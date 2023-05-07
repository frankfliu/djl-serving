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

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDList;
import ai.djl.translate.ServingTranslator;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.util.Map;

/** A {@link Translator} that can handle text generation task. */
public class FasterTransformerServingTranslator implements ServingTranslator {

    Translator<String, String> translator;

    /**
     * Constructs a new {@code FasterTransformerServingTranslator} instance.
     *
     * @param translator the {@link FasterTransformerTranslator}
     */
    public FasterTransformerServingTranslator(Translator<String, String> translator) {
        this.translator = translator;
    }

    /** {@inheritDoc} */
    @Override
    public void setArguments(Map<String, ?> arguments) {}

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Input input) throws Exception {
        String text = input.getAsString(0);
        return translator.processInput(ctx, text);
    }

    /** {@inheritDoc} */
    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) throws Exception {
        Output out = new Output();
        out.add(translator.processOutput(ctx, list));
        return out;
    }
}
