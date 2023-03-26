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

class ModelMetadata {

    String name;
    String[] versions;
    String platform;
    DataDescriptor[] inputs;
    DataDescriptor[] outputs;

    public void setName(String name) {
        this.name = name;
    }

    public void setVersions(String[] versions) {
        this.versions = versions;
    }

    public void setPlatform(String platform) {
        this.platform = platform;
    }

    public void setInputs(DataDescriptor[] inputs) {
        this.inputs = inputs;
    }

    public void setOutputs(DataDescriptor[] outputs) {
        this.outputs = outputs;
    }

    public DataDescriptor getInput(String name) {
        for (DataDescriptor dd : inputs) {
            if (dd.name.equals(name)) {
                return dd;
            }
        }
        return null;
    }

    public DataDescriptor getOutput(String name) {
        for (DataDescriptor dd : outputs) {
            if (dd.name.equals(name)) {
                return dd;
            }
        }
        return null;
    }
}
