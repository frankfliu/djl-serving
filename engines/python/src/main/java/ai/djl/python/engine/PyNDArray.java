/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.python.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrayAdapter;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import java.nio.ByteBuffer;
import java.util.UUID;

/** {@code PyNDArray} is the Python engine of {@link NDArray}. */
public class PyNDArray extends NDArrayAdapter {

    private ByteBuffer data;

    PyNDArray(
            PyNDManager manager,
            NDManager alternativeManager,
            ByteBuffer buffer,
            Shape shape,
            DataType dataType) {
        super(manager, alternativeManager, shape, dataType, UUID.randomUUID().toString());
        this.data = buffer;
        manager.attachInternal(uid, this);
    }

    /** {@inheritDoc} */
    @Override
    public void intern(NDArray replaced) {
        data = ((PyNDArray) replaced).data;
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = PyNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        data.rewind();
        return data;
    }
}
