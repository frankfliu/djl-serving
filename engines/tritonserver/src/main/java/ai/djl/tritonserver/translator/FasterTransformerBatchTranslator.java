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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;

/** The translator for FasterTransformer text generation model. */
public class FasterTransformerBatchTranslator implements NoBatchifyTranslator<String[], String[]> {

    private FasterTransformerTranslator translator;
    private HuggingFaceTokenizer tokenizer;
    private Batchifier batchifier;

    FasterTransformerBatchTranslator(
            FasterTransformerTranslator translator,
            HuggingFaceTokenizer tokenizer,
            Batchifier batchifier) {
        this.translator = translator;
        this.tokenizer = tokenizer;
        this.batchifier = batchifier;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String[] inputs) {
        NDManager manager = ctx.getNDManager();
        Encoding[] encodings = tokenizer.batchEncode(inputs);
        NDList[] batch = new NDList[encodings.length];
        for (int i = 0; i < encodings.length; ++i) {
            batch[i] = encodings[i].toNDList(manager, false);
        }
        return batchifier.batchify(batch);
    }

    /** {@inheritDoc} */
    @Override
    public String[] processOutput(TranslatorContext ctx, NDList list) throws TranslateException {
        NDList[] batch = batchifier.unbatchify(list);
        String[] ret = new String[batch.length];
        for (int i = 0; i < batch.length; ++i) {
            ret[i] = translator.processOutput(ctx, batch[i]);
        }
        return ret;
    }
}
