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
import ai.djl.inference.streaming.ChunkedBytesSupplier;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.bytedeco.tritonserver.global.tritonserver;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_InferenceRequest;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_InferenceResponse;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_ResponseAllocator;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.TimeUnit;

class StreamResponse extends ChunkedBytesSupplier {

    private ModelMetadata metadata;
    private NDManager manager;
    private TRITONSERVER_InferenceRequest req;
    private TRITONSERVER_ResponseAllocator allocator;
    private TranslatorContext context;
    private Translator<?, ?> translator;

    public StreamResponse(
            ModelMetadata metadata,
            NDManager manager,
            TRITONSERVER_InferenceRequest req,
            TRITONSERVER_ResponseAllocator allocator) {
        this.metadata = metadata;
        this.manager = manager;
        this.req = req;
        this.allocator = allocator;
    }

    public void setContext(TranslatorContext context, Translator<?, ?> translator) {
        this.context = context;
        this.translator = translator;
    }

    public void appendContent(TRITONSERVER_InferenceResponse response, boolean lastChunk) {
        NDList data = toNDList(response);
        super.appendContent(data, lastChunk);
        if (lastChunk) {
            JniUtils.checkCall(tritonserver.TRITONSERVER_InferenceResponseDelete(response));
            JniUtils.checkCall(tritonserver.TRITONSERVER_InferenceRequestDelete(req));
            JniUtils.checkCall(tritonserver.TRITONSERVER_ResponseAllocatorDelete(allocator));
        }
    }

    /** {@inheritDoc} */
    @Override
    public byte[] nextChunk(long timeout, TimeUnit unit) throws InterruptedException {
        if (translator != null) {
            NDList list = (NDList) next(timeout, unit);
            try {
                Output o = (Output) translator.processOutput(context, list);
                return o.getData().getAsBytes();
            } catch (Exception e) {
                throw new IllegalStateException(e);
            }
        }
        return super.nextChunk(timeout, unit);
    }

    NDList toNDList(TRITONSERVER_InferenceResponse resp) {
        int[] outputCount = {0};
        JniUtils.checkCall(
                tritonserver.TRITONSERVER_InferenceResponseOutputCount(resp, outputCount));
        NDList list = new NDList(outputCount[0]);

        for (int i = 0; i < outputCount[0]; ++i) {
            BytePointer cname = new BytePointer((Pointer) null);
            IntPointer datatype = new IntPointer(1);
            LongPointer shape = new LongPointer((Pointer) null);
            LongPointer dimCount = new LongPointer(1);
            Pointer base = new Pointer();
            SizeTPointer byteSize = new SizeTPointer(1);
            IntPointer memoryType = new IntPointer(1);
            LongPointer memoryTypeId = new LongPointer(1);
            Pointer userPtr = new Pointer();

            JniUtils.checkCall(
                    tritonserver.TRITONSERVER_InferenceResponseOutput(
                            resp,
                            i,
                            cname,
                            datatype,
                            shape,
                            dimCount,
                            base,
                            byteSize,
                            memoryType,
                            memoryTypeId,
                            userPtr));

            if (cname.isNull()) {
                throw new EngineException("Unable to get output name.");
            }
            String name = cname.getString();

            DataDescriptor dd = metadata.getOutput(name);
            if (dd == null) {
                throw new EngineException("Unexpected output name: " + name);
            }
            int size = Math.toIntExact(dimCount.get());
            long[] s = new long[size];
            shape.get(s);
            TsDataType type = TsDataType.values()[datatype.get()];
            if (type != dd.datatype) {
                throw new EngineException("Unexpected datatype " + type + " for " + name);
            }
            int len = Math.toIntExact(byteSize.get());
            byte[] buf = new byte[len];
            base.limit(len).asByteBuffer().get(buf);
            ByteBuffer bb = ByteBuffer.wrap(buf);
            bb.order(ByteOrder.nativeOrder());
            NDArray array = manager.create(bb, new Shape(s), type.toDataType());
            array.setName(dd.name);
            list.add(array);
        }
        return list;
    }
}
