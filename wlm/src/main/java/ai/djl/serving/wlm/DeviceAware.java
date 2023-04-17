/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.serving.wlm;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.engine.StandardCapabilities;
import ai.djl.util.NeuronUtils;
import ai.djl.util.Utils;
import ai.djl.util.cuda.CudaUtils;

import java.lang.management.MemoryUsage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

class DeviceAware {

    private static final Pattern PATTERN = Pattern.compile("\\{(\\d+)}");

    private static final int[] AVAILABLE_DEVICES = getAvailableDevices();
    private static final boolean HAS_NEURON = NeuronUtils.hasNeuron();
    private static int maxSharedDevice;

    private ModelInfo<?, ?> modelInfo;
    private Device[] devices;
    private int numSlot;
    private boolean mpiMode;
    private boolean exclusive;
    private int[] assignedDevice;

    DeviceAware(ModelInfo<?, ?> modelInfo, Device[] devices, boolean mpiMode, boolean exclusive) {
        this.modelInfo = modelInfo;
        this.devices = devices;
        this.mpiMode = mpiMode;
        this.exclusive = exclusive;
    }

    DeviceAware(ModelInfo<?, ?> modelInfo, int numSlot, boolean mpiMode, boolean exclusive) {
        this.modelInfo = modelInfo;
        this.numSlot = numSlot;
        this.mpiMode = mpiMode;
        this.exclusive = exclusive;
    }

    static DeviceAware parse(ModelInfo<?, ?> modelInfo) {
        String loadOnDevices = modelInfo.loadOnDevices;
        String engineName = modelInfo.engineName;
        Engine engine = Engine.getEngine(modelInfo.engineName);
        if (AVAILABLE_DEVICES.length == 0
                || loadOnDevices.isEmpty()
                || (!HAS_NEURON && !engine.hasCapability(StandardCapabilities.CUDA))) {
            return new DeviceAware(modelInfo, 0, false, false);
        }

        boolean mpiMode = "DeepSpeed".equals(engineName) || "FasterTransformer".equals(engineName);
        boolean exclusive;
        if (loadOnDevices.endsWith("-")) {
            loadOnDevices = loadOnDevices.substring(0, loadOnDevices.length() - 1);
            exclusive = true;
        } else {
            exclusive = tp > 1 || mpiMode || (HAS_NEURON && "Python".equals(engineName));
        }
        Matcher matcher = PATTERN.matcher(loadOnDevices);
        if (matcher.matches()) {
            int sumSlot = Integer.parseInt(matcher.group(1));
            return new DeviceAware(modelInfo, sumSlot, mpiMode, exclusive);
        } else if ("*".equals(loadOnDevices)) {
            return new DeviceAware(modelInfo, -1, mpiMode, exclusive);
        } else {
            String[] token = loadOnDevices.split(";");
            Device[] dev = new Device[token.length];
            for (int i = 0; i < token.length; ++i) {
                dev[i] = Device.fromName(token[i]);
            }
            return new DeviceAware(modelInfo, dev, mpiMode, exclusive);
        }
    }

    String[] allocateDevices() {
        int[] slotIds;
        if (devices == null) {
            if (numSlot == 0) {
                // CPU
                return new String[] {"-1"};
            }
            if (exclusive) {
                slotIds = findNextExclusiveSlot();
            } else {
                slotIds = findNextSharedSlot();
            }
        } else {
            slotIds = Arrays.stream(devices).mapToInt(Device::getDeviceId).toArray();
        }

        int devicePerSlot;
        if (mpiMode) {
            devicePerSlot = modelInfo.getTensorParallelDegree() * modelInfo.getMaxWorkers();
        } else {
            devicePerSlot = modelInfo.getTensorParallelDegree();
        }

        assignedDevice = assignDevice(slotIds, devicePerSlot);

        return Arrays.stream(devices).map(Object::toString).toArray(String[]::new);
    }

    private int[] assignDevice(int[] deviceIds, int total) {
        int[] assignment = new int[AVAILABLE_DEVICES.length];
        for (int deviceId : deviceIds) {
            for (int i = deviceId; i < total; ++i) {
                if (i >= AVAILABLE_DEVICES.length || assignment[i] == 1) {
                    throw new EngineException(
                            "GPU devices are not enough, requested: "
                                    + total
                                    + ", available: "
                                    + AVAILABLE_DEVICES.length);
                }
                assignment[i] = 1;
            }
        }
        return assignment;
    }

    private boolean isSlotAvailable(int slot, int total) {
        for (int i = 0; i < total; ++i) {
            int deviceId = slot * total + i;
            if (deviceId >= AVAILABLE_DEVICES.length) {
                return false;
            }
            if (exclusive) {
                if (AVAILABLE_DEVICES[deviceId] != 0) {
                    return false;
                }
            } else {
                if (AVAILABLE_DEVICES[deviceId] == 1) {
                    return false;
                }
            }
        }
        return true;
    }

    private int[] findNextExclusiveSlot() {
        int devicePerSlot;
        if (mpiMode) {
            devicePerSlot = modelInfo.getTensorParallelDegree() * modelInfo.getMaxWorkers();
        } else {
            devicePerSlot = modelInfo.getTensorParallelDegree();
        }
        int totalSlot;
        if (maxSharedDevice == -1) {
            totalSlot = AVAILABLE_DEVICES.length / devicePerSlot;
        } else {
            totalSlot = (AVAILABLE_DEVICES.length - maxSharedDevice) / devicePerSlot;
        }
        List<Integer> availableSlots = new ArrayList<>();
        for (int i = 0; i < totalSlot; ++i) {
            if (isSlotAvailable(i, devicePerSlot)) {
                availableSlots.add(i);
            }
        }
        if (availableSlots.isEmpty()) {
            throw new EngineException("no available slots found");
        }
        if (numSlot != -1 && availableSlots.size() < numSlot) {
            throw new EngineException("no enough slots found");
        }
        return availableSlots.stream().mapToInt(Integer::intValue).toArray();
    }

    private int[] findNextSharedSlot() {
        // 1 device per slot, cannot be mpi mode
        int totalSlot;
        if (maxSharedDevice == -1) {
            totalSlot = AVAILABLE_DEVICES.length;
        } else {
            totalSlot = maxSharedDevice;
        }
        List<Integer> availableSlots = new ArrayList<>();
        // search backward for shared devices
        for (int i = 0; i < totalSlot; ++i) {
            int index = AVAILABLE_DEVICES.length - 1 - i;
            if (isSlotAvailable(index, 1)) {
                availableSlots.add(i);
            }
        }
        if (availableSlots.isEmpty()) {
            throw new EngineException("no available shared slots found");
        }

        if (numSlot != -1) {
            if (availableSlots.size() < numSlot) {
                throw new EngineException("no enough shared slots found");
            }
            // sort availableSlots by GPU memory
            availableSlots = availableSlots.subList(0, numSlot);
        }
        return availableSlots.stream().mapToInt(Integer::intValue).toArray();
    }

    private static synchronized boolean acquireExclusiveDevice(int deviceId, int total) {
        if (deviceId + total > AVAILABLE_DEVICES.length) {
            return false;
        }
        for (int i = deviceId; i < deviceId + total; ++i) {
            if (AVAILABLE_DEVICES[i] != 0) {
                return false;
            }
        }
        for (int i = deviceId; i < deviceId + total; ++i) {
            AVAILABLE_DEVICES[i] = 1;
        }
        return true;
    }

    private static synchronized boolean acquireExclusiveDevice(int total) {
        if (total > AVAILABLE_DEVICES.length) {
            return false;
        }
        for (int i = 0; i < AVAILABLE_DEVICES.length - total; ++i) {
            if (acquireDeviceExclusive(i, total)) {
                return true;
            }
        }
        return false;
    }

    private static synchronized boolean acquireSharedDevice(
            int deviceId, long requiredMemory, long reservedMemory) {
        if (AVAILABLE_DEVICES[deviceId] != 2) {
            return false;
        }
        // FIXME: Assume is GPU
        MemoryUsage mem = CudaUtils.getGpuMemory(Device.gpu(deviceId));
        long free = mem.getMax() - mem.getCommitted();
        return free - requiredMemory > reservedMemory;
    }

    private static int getMaxSharedDevice() {
        String sharedDevices = Utils.getEnvOrSystemProperty("SERVING_SHARED_DEVICES");
        if (sharedDevices == null) {
            return AVAILABLE_DEVICES.length;
        }
        float ratio = Float.parseFloat(sharedDevices);
        if (ratio > 1) {
            return (int) ratio;
        }
        return (int) (AVAILABLE_DEVICES.length * ratio);
    }

    private static int[] getAvailableDevices() {
        int gpuCount = CudaUtils.getGpuCount();
        int[] devices;
        if (gpuCount > 0) {
            devices = new int[gpuCount];
        } else {
            devices = new int[NeuronUtils.getNeuronCores()];
        }
        maxSharedDevice = getMaxSharedDevice();
        return devices;
    }
}
