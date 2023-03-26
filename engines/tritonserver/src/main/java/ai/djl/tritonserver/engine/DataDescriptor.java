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

import ai.djl.ndarray.types.DataType;
import ai.djl.translate.ArgumentsUtil;

import java.util.Map;

class DataDescriptor {

    String name;
    TsDataType datatype;
    long[] shape;
    Map<String, Object> parameters;

    public void setName(String name) {
        this.name = name;
    }

    public void setDatatype(TsDataType datatype) {
        this.datatype = datatype;
    }

    public void setShape(long[] shape) {
        this.shape = shape;
    }

    public void setParameters(Map<String, Object> parameters) {
        this.parameters = parameters;
    }

    public boolean isBinaryData() {
        if (parameters == null) {
            return false;
        }
        return ArgumentsUtil.booleanValue(parameters, "binary_data");
    }

    public int getBinaryDataSize() {
        if (parameters == null) {
            return -1;
        }
        return ArgumentsUtil.intValue(parameters, "binary_data_size", -1);
    }

    public boolean dataTypeNotEquals(DataType other) {
        return datatype.toDataType() != other;
    }

    public boolean shapeNotEquals(long[] other) {
        if (shape.length != other.length) {
            return true;
        }
        for (int i = 0; i < shape.length; ++i) {
            if (shape[i] != other[i] && shape[i] != -1 && other[i] != -1) {
                return true;
            }
        }
        return false;
    }
}
