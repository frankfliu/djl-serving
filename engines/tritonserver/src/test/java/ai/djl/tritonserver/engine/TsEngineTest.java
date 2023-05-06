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
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
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
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.ServingTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterSuite;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;
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
    public void testServingTritonModel() throws ModelException, IOException, TranslateException {
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(Paths.get("/opt/ml/model/fastertransformer"))
                        .optTranslator(new FasterTransformerServingTranslator())
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
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test(enabled = false)
    public void testTritonModel() throws ModelException, IOException, TranslateException {
        Criteria<String, String> criteria =
                Criteria.builder()
                        .setTypes(String.class, String.class)
                        .optModelPath(Paths.get("/opt/ml/model/fastertransformer"))
                        .optTranslator(new FasterTransformerTranslator())
                        .optEngine("TritonServer")
                        .build();

        try (ZooModel<String, String> model = criteria.loadModel();
                Predictor<String, String> predictor = model.newPredictor()) {
            String output =
                    predictor.predict("translate English to German: The house is wonderful.");
            logger.info("========== {}", output);
            Assert.assertEquals(output, "Das Haus ist wunderbar.</s>");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test(enabled = false)
    public void testNDListInput() throws ModelException, IOException, TranslateException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelPath(Paths.get("/opt/ml/model/simple"))
                        .optOption("model_loading_timeout", "1")
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

    private static final class FasterTransformerServingTranslator implements ServingTranslator {

        FasterTransformerTranslator translator;

        public FasterTransformerServingTranslator() {
            translator = new FasterTransformerTranslator();
        }

        @Override
        public void setArguments(Map<String, ?> arguments) {}

        @Override
        public NDList processInput(TranslatorContext ctx, Input input) {
            String text = input.getAsString(0);
            return translator.processInput(ctx, text);
        }

        @Override
        public Output processOutput(TranslatorContext ctx, NDList list) throws Exception {
            Output out = new Output();
            out.add(translator.processOutput(ctx, list));
            return out;
        }
    }

    private static final class FasterTransformerTranslator
            implements NoBatchifyTranslator<String, String> {

        private HuggingFaceTokenizer tokenizer;

        public FasterTransformerTranslator() {
            tokenizer = HuggingFaceTokenizer.newInstance("google/flan-t5-xl");
        }

        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            Encoding encoding = tokenizer.encode(input);
            long[] ids = encoding.getIds();
            int[] inputIds = Arrays.stream(ids).boxed().mapToInt(Long::intValue).toArray();
            return createInput(ctx.getNDManager(), inputIds);
        }

        @Override
        public String processOutput(TranslatorContext ctx, NDList list) throws TranslateException {
            Integer start = (Integer) ctx.getAttachment("start");
            if (start == null) {
                start = 0;
            }
            NDArray array = list.get("output_ids");
            NDArray sequenceLength = list.get("sequence_length");
            int outSize = sequenceLength.toIntArray()[0] - start - 1;
            int[] buf = array.toIntArray();
            if (outSize > buf.length) {
                throw new TranslateException("Invalid sequence_length: " + outSize);
            }
            long[] ids = new long[outSize];
            for (int i = start; i < outSize; ++i) {
                ids[i] = buf[i];
            }
            ctx.setAttachment("start", outSize);
            return tokenizer.decode(ids).trim();
        }

        private NDList createInput(NDManager manager, int[] inputIds) {
            ByteBuffer bb = manager.allocateDirect(inputIds.length * 4);
            bb.asIntBuffer().put(inputIds);
            bb.rewind();
            NDArray input0 = manager.create(bb, new Shape(1, inputIds.length), DataType.UINT32);
            input0.setName("input_ids");
            bb = manager.allocateDirect(4);
            bb.putInt(0, inputIds.length);
            NDArray input1 = manager.create(bb, new Shape(1, 1), DataType.UINT32);
            input1.setName("sequence_length");
            bb = manager.allocateDirect(4);
            bb.putInt(0, 127);
            NDArray input2 = manager.create(bb, new Shape(1, 1), DataType.UINT32);
            input2.setName("max_output_len");
            return new NDList(input0, input1, input2);
        }
    }
}
