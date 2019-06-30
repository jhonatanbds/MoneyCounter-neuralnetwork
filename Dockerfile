# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG TF_SERVING_VERSION=1.11.1
ARG TF_SERVING_BUILD_IMAGE=tensorflow/serving:${TF_SERVING_VERSION}-devel

FROM ${TF_SERVING_BUILD_IMAGE} as build_image
FROM ubuntu:16.04

ARG TF_SERVING_VERSION_GIT_BRANCH=r1.11
ARG TF_SERVING_VERSION_GIT_COMMIT=head

LABEL maintainer="mohammed.alnakli@gmail.com"
LABEL tensorflow_serving_github_branchtag=${TF_SERVING_VERSION_GIT_BRANCH}
LABEL tensorflow_serving_github_commit=${TF_SERVING_VERSION_GIT_COMMIT}

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install TF Serving pkg
COPY --from=build_image /usr/local/bin/tensorflow_model_server /usr/bin/tensorflow_model_server

# Expose is NOT supported by Heroku

# Expose ports
# gRPC
# EXPOSE 8500

# REST
# EXPOSE 8501

# Set where models should be stored in the container
ENV MODEL_BASE_PATH=/models
RUN mkdir -p ${MODEL_BASE_PATH}
WORKDIR  ${MODEL_BASE_PATH}
# The only required piece is the model name in order to differentiate endpoints
ENV MODEL_NAME=model_rpn

COPY models/model_rpn.h5 model_rpn

# Create a script that runs the model server so we can use environment variables
# while also passing in arguments from the docker command line

# $PORT is set by Heroku			
RUN echo '#!/bin/bash \n\n\
tensorflow_model_server  --rest_api_port=$PORT \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh
# CMD is required to run on Heroku
CMD ["/usr/bin/tf_serving_entrypoint.sh"]