# Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_library(FortranEvaluateTesting
  testing.cc
  fp-testing.cc
)

add_executable(leading-zero-bit-count-test
  leading-zero-bit-count.cc
)

target_link_libraries(leading-zero-bit-count-test
  FortranEvaluateTesting
  FortranEvaluate
)

add_executable(bit-population-count-test
  bit-population-count.cc
)

target_link_libraries(bit-population-count-test
  FortranEvaluateTesting
  FortranEvaluate
)

add_executable(expression-test
  expression.cc
)

target_link_libraries(expression-test
  FortranEvaluateTesting
  FortranSemantics
  FortranEvaluate
  FortranParser
)

add_executable(integer-test
  integer.cc
)

target_link_libraries(integer-test
  FortranEvaluateTesting
  FortranEvaluate
)

add_executable(intrinsics-test
  intrinsics.cc
)

target_link_libraries(intrinsics-test
  FortranEvaluateTesting
  FortranSemantics
  FortranEvaluate
  FortranParser
  FortranRuntime
)

add_executable(logical-test
  logical.cc
)

target_link_libraries(logical-test
  FortranEvaluateTesting
  FortranEvaluate
)

add_executable(real-test
  real.cc
)

target_link_libraries(real-test
  FortranEvaluateTesting
  FortranEvaluate
  m
)

add_executable(reshape-test
  reshape.cc
)

target_link_libraries(reshape-test
  FortranEvaluateTesting
  FortranSemantics
  FortranEvaluate
  FortranRuntime
)

add_executable(ISO-Fortran-binding-test
  ISO-Fortran-binding.cc
)

target_link_libraries(ISO-Fortran-binding-test
  FortranEvaluateTesting
  FortranEvaluate
  FortranRuntime
)

add_executable(folding-test
  folding.cc
)

target_link_libraries(folding-test
  FortranEvaluateTesting
  FortranEvaluate
)

add_test(Expression expression-test)
add_test(Leadz leading-zero-bit-count-test)
add_test(PopPar bit-population-count-test)
add_test(Integer integer-test)
add_test(Intrinsics intrinsics-test)
add_test(Logical logical-test)
add_test(Real real-test)
add_test(RESHAPE reshape-test)
add_test(ISO-binding ISO-Fortran-binding-test)
add_test(folding folding-test)
