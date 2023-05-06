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
package ai.djl.tritonserver.engine;

import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.bytedeco.tritonserver.global.tritonserver;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_Error;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_InferenceRequest;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_InferenceRequestReleaseFn_t;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_InferenceResponse;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_InferenceResponseCompleteFn_t;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_Message;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_ResponseAllocator;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_ResponseAllocatorAllocFn_t;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_ResponseAllocatorReleaseFn_t;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_Server;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_ServerOptions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A class containing utilities to interact with the TritonServer C API. */
@SuppressWarnings("MissingJavadocMethod")
public final class JniUtils {

    private static final Logger logger = LoggerFactory.getLogger(JniUtils.class);

    private static final Map<Pointer, StreamResponse> RESPONSES = new ConcurrentHashMap<>();
    private static final ResponseAlloc RESP_ALLOC = new ResponseAlloc();
    private static final ResponseRelease RESP_RELEASE = new ResponseRelease();
    private static final InferRequestComplete REQUEST_COMPLETE = new InferRequestComplete();
    private static final InferResponseComplete RESPONSE_COMPLETE = new InferResponseComplete();

    private JniUtils() {}

    public static TRITONSERVER_Server initTritonServer() {
        // TODO: Use TRITONSERVER_ServerRegisterModelRepository after triton engine initialized
        String modelStore = Utils.getEnvOrSystemProperty("SERVING_MODEL_STORE");
        if (modelStore == null || modelStore.isEmpty()) {
            modelStore = "/opt/ml/model";
        }
        int[] major = {0};
        int[] minor = {0};
        checkCall(tritonserver.TRITONSERVER_ApiVersion(major, minor));
        if ((tritonserver.TRITONSERVER_API_VERSION_MAJOR != major[0])
                || (tritonserver.TRITONSERVER_API_VERSION_MINOR > minor[0])) {
            throw new EngineException("triton API version mismatch");
        }

        TRITONSERVER_ServerOptions options = new TRITONSERVER_ServerOptions(null);
        checkCall(tritonserver.TRITONSERVER_ServerOptionsNew(options));
        checkCall(
                tritonserver.TRITONSERVER_ServerOptionsSetModelRepositoryPath(options, modelStore));
        String verboseLevel = Utils.getEnvOrSystemProperty("TRITON_VERBOSE_LEVEL");
        int verbosity;
        if (verboseLevel != null) {
            verbosity = Integer.parseInt(verboseLevel);
        } else {
            verbosity = logger.isTraceEnabled() ? 1 : 0;
        }
        checkCall(tritonserver.TRITONSERVER_ServerOptionsSetLogVerbose(options, verbosity));
        checkCall(
                tritonserver.TRITONSERVER_ServerOptionsSetBackendDirectory(
                        options, "/opt/tritonserver/backends"));
        checkCall(
                tritonserver.TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
                        options, "/opt/tritonserver/repoagents"));
        checkCall(tritonserver.TRITONSERVER_ServerOptionsSetStrictModelConfig(options, true));
        checkCall(
                tritonserver.TRITONSERVER_ServerOptionsSetModelControlMode(
                        options, tritonserver.TRITONSERVER_MODEL_CONTROL_EXPLICIT));

        // TODO: Stop triton and delete triton. Currently TritonServer will live for ever
        TRITONSERVER_Server triton = new TRITONSERVER_Server(null);
        checkCall(tritonserver.TRITONSERVER_ServerNew(triton, options));
        checkCall(tritonserver.TRITONSERVER_ServerOptionsDelete(options));

        // Wait until the triton is both live and ready.
        for (int i = 0; i < 10; ++i) {
            boolean[] live = {false};
            boolean[] ready = {false};
            checkCall(tritonserver.TRITONSERVER_ServerIsLive(triton, live));
            checkCall(tritonserver.TRITONSERVER_ServerIsReady(triton, ready));
            logger.debug("Triton health: live {}, ready: {}", live[0], ready[0]);
            if (live[0] && ready[0]) {
                printServerStatus(triton);
                return triton;
            }
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                throw new EngineException("Triton startup interrupted.", e);
            }
        }
        throw new EngineException("Failed to find triton healthy.");
    }

    public static void loadModel(TRITONSERVER_Server triton, String modelName, int timeout) {
        checkCall(tritonserver.TRITONSERVER_ServerLoadModel(triton, modelName));

        // Wait for the model to become available.
        boolean[] ready = {false};
        while (!ready[0]) {
            checkCall(tritonserver.TRITONSERVER_ServerModelIsReady(triton, modelName, 1, ready));
            if (ready[0]) {
                break;
            }

            if (timeout < 0) {
                throw new EngineException("Model loading timed out in: " + timeout);
            }

            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                throw new EngineException("Model loading interrupted.", e);
            }
            timeout -= 500;
        }
    }

    public static void unloadModel(TRITONSERVER_Server triton, String modelName, int timeout) {
        tritonserver.TRITONSERVER_ServerUnloadModel(triton, modelName);
        // TODO: wait model fully unloaded
    }

    public static ModelMetadata getModelMetadata(TRITONSERVER_Server triton, String modelName) {
        TRITONSERVER_Message metadata = new TRITONSERVER_Message(null);
        checkCall(tritonserver.TRITONSERVER_ServerModelMetadata(triton, modelName, 1, metadata));
        BytePointer buffer = new BytePointer((Pointer) null);
        SizeTPointer size = new SizeTPointer(1);
        checkCall(tritonserver.TRITONSERVER_MessageSerializeToJson(metadata, buffer, size));
        String json = buffer.limit(size.get()).getString();
        checkCall(tritonserver.TRITONSERVER_MessageDelete(metadata));
        return JsonUtils.GSON.fromJson(json, ModelMetadata.class);
    }

    public static StreamResponse predict(
            TRITONSERVER_Server triton, ModelMetadata metadata, NDManager manager, NDList input) {
        TRITONSERVER_InferenceRequest req = toRequest(triton, metadata, input);
        return predict(triton, metadata, manager, req);
    }

    private static StreamResponse predict(
            TRITONSERVER_Server triton,
            ModelMetadata metadata,
            NDManager manager,
            TRITONSERVER_InferenceRequest req) {
        // TODO: Can this allocator be re-used?
        TRITONSERVER_ResponseAllocator allocator = new TRITONSERVER_ResponseAllocator(null);
        checkCall(
                tritonserver.TRITONSERVER_ResponseAllocatorNew(
                        allocator, RESP_ALLOC, RESP_RELEASE, null));

        for (DataDescriptor dd : metadata.outputs) {
            checkCall(tritonserver.TRITONSERVER_InferenceRequestAddRequestedOutput(req, dd.name));
        }

        // Perform inference...
        StreamResponse sr = new StreamResponse(metadata, manager, req, allocator);
        RESPONSES.put(req, sr);

        checkCall(
                tritonserver.TRITONSERVER_InferenceRequestSetResponseCallback(
                        req, allocator, null, RESPONSE_COMPLETE, req));

        checkCall(tritonserver.TRITONSERVER_ServerInferAsync(triton, req, null));
        return sr;
    }

    private static TRITONSERVER_InferenceRequest toRequest(
            TRITONSERVER_Server triton, ModelMetadata metadata, NDList list) {
        TRITONSERVER_InferenceRequest req = new TRITONSERVER_InferenceRequest(null);
        checkCall(tritonserver.TRITONSERVER_InferenceRequestNew(req, triton, metadata.name, -1));

        // TODO: REQUEST_ID
        checkCall(tritonserver.TRITONSERVER_InferenceRequestSetId(req, "my_request_id"));
        checkCall(
                tritonserver.TRITONSERVER_InferenceRequestSetReleaseCallback(
                        req, REQUEST_COMPLETE, null));

        for (NDArray array : list) {
            String name = array.getName();
            if (name == null) {
                throw new IllegalArgumentException("input name is required.");
            }
            TsDataType dtype = TsDataType.fromDataType(array.getDataType());
            long[] shape = array.getShape().getShape();
            DataDescriptor dd = metadata.getInput(name);
            if (dd.datatype != dtype) {
                throw new IllegalArgumentException(
                        "Invalid input data type " + dtype + ", expected: " + dd.datatype);
            }

            byte[] buf = array.toByteArray();
            BytePointer bp = new BytePointer(buf);

            checkCall(
                    tritonserver.TRITONSERVER_InferenceRequestAddInput(
                            req, dd.name, dd.datatype.ordinal(), shape, shape.length));
            // input/output always copy to CPU for now
            checkCall(
                    tritonserver.TRITONSERVER_InferenceRequestAppendInputData(
                            req, dd.name, bp, buf.length, tritonserver.TRITONSERVER_MEMORY_CPU, 0));
        }
        return req;
    }

    private static void printServerStatus(TRITONSERVER_Server triton) {
        // Print status of the triton.
        TRITONSERVER_Message metadata = new TRITONSERVER_Message(null);
        checkCall(tritonserver.TRITONSERVER_ServerMetadata(triton, metadata));
        BytePointer buffer = new BytePointer((Pointer) null);
        SizeTPointer size = new SizeTPointer(1);
        checkCall(tritonserver.TRITONSERVER_MessageSerializeToJson(metadata, buffer, size));

        logger.info("Server Status: {}", buffer.limit(size.get()).getString());
        checkCall(tritonserver.TRITONSERVER_MessageDelete(metadata));
    }

    static void checkCall(TRITONSERVER_Error err) {
        if (err != null) {
            String error =
                    tritonserver.TRITONSERVER_ErrorCodeString(err)
                            + " - "
                            + tritonserver.TRITONSERVER_ErrorMessage(err);
            tritonserver.TRITONSERVER_ErrorDelete(err);
            throw new EngineException(error);
        }
    }

    private static final class ResponseAlloc extends TRITONSERVER_ResponseAllocatorAllocFn_t {

        /** {@inheritDoc} */
        @Override
        @SuppressWarnings("rawtypes")
        public TRITONSERVER_Error call(
                TRITONSERVER_ResponseAllocator allocator,
                String tensorName,
                long byteSize,
                int preferredMemoryType,
                long preferredMemoryTypeId,
                Pointer userPtr,
                PointerPointer buffer,
                PointerPointer bufferUserPtr,
                IntPointer actualMemoryType,
                LongPointer actualMemoryTypeId) {
            actualMemoryType.put(0, tritonserver.TRITONSERVER_MEMORY_CPU);
            actualMemoryTypeId.put(0, 0);

            bufferUserPtr.put(0, null);
            if (byteSize == 0) {
                buffer.put(0, null);
            } else {
                Pointer allocatedPtr = Pointer.malloc(byteSize);
                if (!allocatedPtr.isNull()) {
                    buffer.put(0, allocatedPtr);
                } else {
                    throw new EngineException("Out of memory, malloc failed.");
                }
            }

            return null;
        }
    }

    private static final class ResponseRelease extends TRITONSERVER_ResponseAllocatorReleaseFn_t {

        /** {@inheritDoc} */
        @Override
        public TRITONSERVER_Error call(
                TRITONSERVER_ResponseAllocator allocator,
                Pointer buffer,
                Pointer bufferUserPtr,
                long byteSize,
                int memoryType,
                long memoryTypeId) {
            Pointer.free(buffer);
            Loader.deleteGlobalRef(bufferUserPtr);
            return null; // Success
        }
    }

    private static final class InferRequestComplete
            extends TRITONSERVER_InferenceRequestReleaseFn_t {

        /** {@inheritDoc} */
        @Override
        public void call(TRITONSERVER_InferenceRequest request, int flags, Pointer userp) {
            // We reuse the request so we don't delete it here.
        }
    }

    private static final class InferResponseComplete
            extends TRITONSERVER_InferenceResponseCompleteFn_t {

        /** {@inheritDoc} */
        @Override
        public void call(TRITONSERVER_InferenceResponse response, int flags, Pointer userp) {
            if (response != null) {
                boolean last = (flags & 1) != 0;
                StreamResponse sr = RESPONSES.get(userp);
                checkCall(tritonserver.TRITONSERVER_InferenceResponseError(response));
                sr.appendContent(response, last);
            }
        }
    }
}
