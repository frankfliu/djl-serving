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

import ai.djl.Device;
import ai.djl.engine.EngineException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_Server;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;

class TsPredictor<I, O> extends Predictor<I, O> {

    private TRITONSERVER_Server triton;
    private ModelMetadata metadata;

    public TsPredictor(
            TRITONSERVER_Server triton, TsModel model, Translator<I, O> translator, Device device) {
        super(model, translator, device, false);
        this.triton = triton;
        this.metadata = model.metadata;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public List<O> batchPredict(List<I> inputs) throws TranslateException {
        I first = inputs.get(0);
        if (!(first instanceof Input)) {
            return super.batchPredict(inputs);
        }

        try (PredictorContext context = new PredictorContext()) {
            if (!prepared) {
                translator.prepare(context);
                prepared = true;
            }
            NDManager ctxMgr = context.getNDManager();

            int size = inputs.size();
            if (size == 1) {
                Input in = (Input) first;
                Output output = new Output();
                if (in.getProperty("Content-Type", "").startsWith("tensor/")) {
                    output.addProperty("Content-Type", "tensor/ndlist");
                    StreamResponse sr =
                            JniUtils.predict(triton, metadata, ctxMgr, in.getDataAsNDList(ctxMgr));
                    output.add(sr);
                } else {
                    NDList tensor = translator.processInput(context, first);
                    StreamResponse sr = JniUtils.predict(triton, metadata, ctxMgr, tensor);
                    sr.setContext(context, translator);
                    output.add(sr);
                }
                return Collections.singletonList((O) output);
            }
        } catch (Exception e) {
            throw new TranslateException(e);
        }

        // TODO: Adds dynamic batching support
        throw new TranslateException("Batch is supported yet.");
    }

    /** {@inheritDoc} */
    @Override
    protected NDList predictInternal(TranslatorContext ctx, NDList ndList) {
        StreamResponse sr = JniUtils.predict(triton, metadata, ctx.getNDManager(), ndList);
        NDList last = null;
        while (sr.hasNext()) {
            try {
                last = (NDList) sr.next(1, TimeUnit.MINUTES);
            } catch (InterruptedException e) {
                throw new EngineException(e);
            }
        }
        return last;
    }
}
