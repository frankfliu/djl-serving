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
package ai.djl.tritonserver.translator;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Batchifier;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Map;

/** The translator for FasterTransformer text generation model. */
public class FasterTransformerTranslator implements NoBatchifyTranslator<String, String> {

    private HuggingFaceTokenizer tokenizer;
    private Batchifier batchifier;

    /**
     * Constructs a new {@code FasterTransformerTranslator} instance.
     *
     * @param tokenizer the Huggingface tokenizer
     * @param batchifier the batchifier
     */
    public FasterTransformerTranslator(HuggingFaceTokenizer tokenizer, Batchifier batchifier) {
        this.tokenizer = tokenizer;
        this.batchifier = batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        Encoding encoding = tokenizer.encode(input);
        long[] ids = encoding.getIds();
        int[] inputIds = Arrays.stream(ids).boxed().mapToInt(Long::intValue).toArray();
        return createInput(ctx.getNDManager(), inputIds);
    }

    /** {@inheritDoc} */
    @Override
    public String processOutput(TranslatorContext ctx, NDList list) throws TranslateException {
        Integer start = (Integer) ctx.getAttachment("start");
        if (start == null) {
            start = 0;
        }
        NDArray array = list.get("output_ids");
        NDArray sequenceLength = list.get("sequence_length");
        int outSize = sequenceLength.toIntArray()[0];
        int[] buf = array.toIntArray();
        if (outSize > buf.length) {
            throw new TranslateException("Invalid sequence_length: " + outSize);
        }
        long[] ids = new long[outSize - start];
        for (int i = 0; i < ids.length; ++i) {
            ids[i] = buf[i + start];
        }
        ctx.setAttachment("start", outSize);
        return tokenizer.decode(ids).trim();
    }

    /** {@inheritDoc} */
    @Override
    public Translator<String[], String[]> toBatchTranslator() {
        return toBatchTranslator(batchifier);
    }

    /** {@inheritDoc} */
    @Override
    public FasterTransformerBatchTranslator toBatchTranslator(Batchifier batchifier) {
        tokenizer.enableBatch();
        return new FasterTransformerBatchTranslator(this, tokenizer, batchifier);
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

    /**
     * Creates a builder to build a {@code FasterTransformerTranslator}.
     *
     * @param tokenizer the tokenizer
     * @return a new builder
     */
    public static Builder builder(HuggingFaceTokenizer tokenizer) {
        return new Builder(tokenizer);
    }

    /**
     * Creates a builder to build a {@code FasterTransformerTranslator}.
     *
     * @param tokenizer the tokenizer
     * @param arguments the models' arguments
     * @return a new builder
     */
    public static Builder builder(HuggingFaceTokenizer tokenizer, Map<String, ?> arguments) {
        Builder builder = builder(tokenizer);
        builder.configure(arguments);

        return builder;
    }

    /** The builder for question answering translator. */
    public static final class Builder {

        private HuggingFaceTokenizer tokenizer;
        private Batchifier batchifier = Batchifier.STACK;

        Builder(HuggingFaceTokenizer tokenizer) {
            this.tokenizer = tokenizer;
        }

        /**
         * Sets the {@link Batchifier} for the {@link Translator}.
         *
         * @param batchifier true to include token types
         * @return this builder
         */
        public Builder optBatchifier(Batchifier batchifier) {
            this.batchifier = batchifier;
            return this;
        }

        /**
         * Configures the builder with the model arguments.
         *
         * @param arguments the model arguments
         */
        public void configure(Map<String, ?> arguments) {
            String batchifierStr = ArgumentsUtil.stringValue(arguments, "batchifier", "stack");
            optBatchifier(Batchifier.fromString(batchifierStr));
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         * @throws IOException if I/O error occurs
         */
        public FasterTransformerTranslator build() throws IOException {
            return new FasterTransformerTranslator(tokenizer, batchifier);
        }
    }
}
