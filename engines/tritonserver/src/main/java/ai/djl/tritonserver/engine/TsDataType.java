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

/** Helper to convert between {@link DataType} an the TritonServer DataTypes. */
public enum TsDataType {
    INVALID,
    BOOL,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    INT8,
    INT16,
    INT32,
    INT64,
    FP16,
    FP32,
    FP64,
    BYTES,
    BF16;

    /**
     * Returns the DJL {@link DataType}.
     *
     * @return the DJL {@code DataType}
     */
    public DataType toDataType() {
        switch (this) {
            case BOOL:
                return DataType.BOOLEAN;
            case UINT8:
                return DataType.UINT8;
            case INT8:
                return DataType.INT8;
            case INT32:
                return DataType.INT32;
            case INT64:
                return DataType.INT64;
            case FP16:
                return DataType.FLOAT16;
            case FP32:
                return DataType.FLOAT32;
            case FP64:
                return DataType.FLOAT64;
            case UINT16:
                return DataType.UINT16;
            case UINT32:
                return DataType.UINT32;
            case UINT64:
                return DataType.UINT64;
            case INT16:
                return DataType.INT16;
            case BF16:
                return DataType.BFLOAT16;
            case BYTES:
                return DataType.STRING;
            case INVALID:
            default:
                return DataType.UNKNOWN;
        }
    }

    /**
     * Converts from DJL {@link DataType}.
     *
     * @param dataType the DJL data type
     * @return the {@code TsDataType}
     */
    public static TsDataType fromDataType(DataType dataType) {
        switch (dataType) {
            case BOOLEAN:
                return TsDataType.BOOL;
            case UINT8:
                return TsDataType.UINT8;
            case INT8:
                return TsDataType.INT8;
            case INT32:
                return TsDataType.INT32;
            case INT64:
                return TsDataType.INT64;
            case FLOAT16:
                return TsDataType.FP16;
            case FLOAT32:
                return TsDataType.FP32;
            case FLOAT64:
                return TsDataType.FP64;
            case STRING:
                return TsDataType.BYTES;
            case UINT16:
                return TsDataType.UINT16;
            case UINT32:
                return TsDataType.UINT32;
            case UINT64:
                return TsDataType.UINT64;
            case INT16:
                return TsDataType.INT16;
            case BFLOAT16:
                return TsDataType.BF16;
            case UNKNOWN:
            default:
                return TsDataType.INVALID;
        }
    }
}
