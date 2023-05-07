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

import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.inference.streaming.ChunkedBytesSupplier;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.tritonserver.translator.FasterTransformerTranslatorFactory;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterSuite;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;

public class TsEngineTest {

    private static final Logger logger = LoggerFactory.getLogger(TsEngineTest.class);

    @AfterSuite
    public void shutdown() {
        logger.info("shutting down");
        TsEngine engine = (TsEngine) Engine.getEngine("TritonServer");
        engine.unload();

        Runtime.getRuntime().halt(0);
    }

    @Test(enabled = true)
    public void testServingTritonModel()
            throws ModelException, IOException, TranslateException, InterruptedException {
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(Paths.get("/opt/ml/model/fastertransformer"))
                        .optArgument("tokenizer", "google/flan-t5-xl")
                        .optTranslatorFactory(new FasterTransformerTranslatorFactory())
                        .optEngine("TritonServer")
                        .build();

        try (ZooModel<Input, Output> model = criteria.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input in = new Input();
            in.add("translate English to German: The house is wonderful.");
            Output output = predictor.predict(in);
            ChunkedBytesSupplier cbs = (ChunkedBytesSupplier) output.getData();
            while (cbs.hasNext()) {
                byte[] line = cbs.nextChunk(1, TimeUnit.MINUTES);
                logger.info("========= {}", new String(line, StandardCharsets.UTF_8));
            }
        }
    }

    @Test(enabled = false)
    public void testTritonModel() throws ModelException, IOException, TranslateException {
        Criteria<String, String> criteria =
                Criteria.builder()
                        .setTypes(String.class, String.class)
                        .optModelPath(Paths.get("/opt/ml/model/fastertransformer"))
                        .optArgument("tokenizer", "google/flan-t5-xl")
                        .optTranslatorFactory(new FasterTransformerTranslatorFactory())
                        .optEngine("TritonServer")
                        .build();

        try (ZooModel<String, String> model = criteria.loadModel();
                Predictor<String, String> predictor = model.newPredictor()) {
            String output =
                    predictor.predict("translate English to German: The house is wonderful.");
            logger.info("========== {}", output);
            Assert.assertEquals(output, "Das Haus ist wunderbar.</s>");
        }
    }

    @Test(enabled = false)
    public void testNDListInput() throws ModelException, IOException, TranslateException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelPath(Paths.get("/opt/ml/model/simple"))
                        .optOption("model_loading_timeout", "1")
                        .optArgument("tokenizer", "google/flan-t5-xl")
                        .optTranslatorFactory(new FasterTransformerTranslatorFactory())
                        .optEngine("TritonServer")
                        .build();

        try (ZooModel<NDList, NDList> model = criteria.loadModel();
                Predictor<NDList, NDList> predictor = model.newPredictor();
                NDManager manager = NDManager.newBaseManager()) {
            NDArray input0 = manager.zeros(new Shape(1, 16), DataType.INT32);
            input0.setName("INPUT0");
            NDArray input1 = manager.ones(new Shape(1, 16), DataType.INT32);
            input1.setName("INPUT1");
            NDList list = new NDList(input0, input1);
            NDList ret = predictor.predict(list);
            Assert.assertEquals(ret.size(), 2);
            Assert.assertEquals(ret.head().getShape(), new Shape(1, 16));
        }
    }
}
