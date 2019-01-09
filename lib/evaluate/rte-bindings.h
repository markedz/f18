// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FORTRAN_EVALUATE_RTE_BINDINGS_H_
#define FORTRAN_EVALUATE_RTE_BINDINGS_H_

// defines signature of rte functions and func pointer for intrinsic folding

#include "common.h"
#include "type.h"
#include "../common/template.h"
#include <array>
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace Fortran::evaluate::rte {

// compute signature to avoid recursive std::visit that would be expensive when
// comparing arg list with a function signature
static_assert(
    std::variant_size_v<SomeHostType> < 256, "Need bigger signature holder");
using SignatureType = unsigned char;
template<size_t N> using SignatureArrayType = std::array<SignatureType, N>;

template<typename T> constexpr SignatureType SignatureFromType() {
  SomeHostType dummy{T{}};
  return dummy.index();
}

template<typename... T> struct SignatureHelper {
  static constexpr int ntypes{sizeof...(T)};

  template<size_t... i>
  static constexpr auto BuildSignatureArray(std::index_sequence<i...>) {
    return SignatureArrayType<sizeof...(i)>{
        {SignatureFromType<std::tuple_element_t<i, std::tuple<T...>>>()...}};
  }

  static constexpr auto BuildSignatureArray() {
    return BuildSignatureArray(std::make_index_sequence<ntypes>{});
  }
};

template<typename TR, typename... TA> class FunctionSymbol {
public:
  using F = TR (*)(TA...);
  static constexpr size_t nargs = sizeof...(TA);
  FunctionSymbol(F f) : handle{f} {};
  TR Call(TA... args) const { return handle(args...); };
  TR Call(std::vector<SomeHostType> &args) const {
    if constexpr (nargs) {
      return CallHelper<nargs - 1>(args);
    } else {
      return handle();
    }
  }
  // std::string SymboleName;
  F handle;
  static constexpr SignatureArrayType<nargs + 1> signature{
      SignatureHelper<TR, TA...>::BuildSignatureArray()};

private:
  template<int pos, typename... TRA>
  constexpr TR CallHelper(
      std::vector<SomeHostType> &args, TRA... resolvedArgs) const {
    static_assert(pos <= nargs);
    if constexpr (pos == 0) {
      return handle(
          std::get<std::tuple_element_t<0, std::tuple<TA...>>>(args[0]),
          resolvedArgs...);
    } else {
      return CallHelper<pos - 1>(args,
          std::get<std::tuple_element_t<pos, std::tuple<TA...>>>(args[pos]),
          resolvedArgs...);
    }
  }
};

// This is a bit annoying, it has to be updated every time a new rte type
// signature is added
// won't compile if not updated
using SomeRteFunType = std::variant<FunctionSymbol<float, float>,
    FunctionSymbol<double, double>, FunctionSymbol<float, float, float>,
    FunctionSymbol<double, double, double>,
    FunctionSymbol<float, signed char, float>,
    FunctionSymbol<float, short, float>, FunctionSymbol<float, int, float>,
    FunctionSymbol<float, long long int, float>,
    FunctionSymbol<double, signed char, double>,
    FunctionSymbol<double, short, double>, FunctionSymbol<double, int, double>,
    FunctionSymbol<double, long long int, double>>;

class RteFunction {
public:
  std::string name;
  SomeRteFunType symbol;
  size_t nargs;
  const SignatureType *signature;

  RteFunction(std::string &&name, SomeRteFunType &&f) : name{name}, symbol{f} {
    std::visit(
        [this](auto &x) {
          this->nargs = x.nargs;
          this->signature = x.signature.data();
        },
        f);
  };
};

template<typename Tn, typename Tx> static Tx bessel_jn(Tn n, Tx x) {
  return std::sph_bessel(n, x);
}

// template void bessel_jn<signed char, float>(signed char n, float x);
// const auto bessel_jn_double_int = bessel_jn<double, int>;
// Defining runtime bindings
static const RteFunction rteFuns[]{
    {{"acos"}, {FunctionSymbol<float, float>{std::acos}}},
    {{"acos"}, {FunctionSymbol<double, double>{std::acos}}},
    {{"asin"}, {FunctionSymbol<float, float>{std::asin}}},
    {{"asin"}, {FunctionSymbol<double, double>{std::asin}}},
    {{"atan2"}, {FunctionSymbol<float, float, float>{std::atan2}}},
    {{"atan2"}, {FunctionSymbol<double, double, double>{std::atan2}}},
    {{"bessel_jn"},
        {FunctionSymbol<float, signed char, float>{
            bessel_jn<signed char, float>}}},
    {{"bessel_jn"},
        {FunctionSymbol<float, short, float>{bessel_jn<short, float>}}},
    {{"bessel_jn"}, {FunctionSymbol<float, int, float>{bessel_jn<int, float>}}},
    {{"bessel_jn"},
        {FunctionSymbol<float, long long int, float>{
            bessel_jn<long long int, float>}}},
    {{"bessel_jn"},
        {FunctionSymbol<double, signed char, double>{
            bessel_jn<signed char, double>}}},
    {{"bessel_jn"},
        {FunctionSymbol<double, short, double>{bessel_jn<short, double>}}},
    {{"bessel_jn"},
        {FunctionSymbol<double, int, double>{bessel_jn<int, double>}}},
    {{"bessel_jn"},
        {FunctionSymbol<double, long long int, double>{
            bessel_jn<long long int, double>}}}};

template<typename T> class RteFunctionRef {
public:
  using Result = T;
  RteFunctionRef(std::string &&name, std::vector<SomeHostType> &&args)
    : name{name}, args{args} {};
  std::string name;
  std::vector<SomeHostType> args;
};

class RuntimeFunctionMap {
public:
  RuntimeFunctionMap() {
    for (const RteFunction &f : rteFuns) {
      functions.insert(std::make_pair(std::string{f.name}, &f));
    }
  }

  template<typename T>
  std::optional<const RteFunction *> find(RteFunctionRef<T> &ref) {
    auto rteFunRange{functions.equal_range(ref.name)};
    constexpr SignatureType resTypeSignature{SignatureFromType<T>()};
    const size_t nargs{ref.args.size()};
    for (auto iter{rteFunRange.first}; iter != rteFunRange.second; ++iter) {
      if (nargs == iter->second->nargs &&
          resTypeSignature == iter->second->signature[0]) {
        bool match{true};
        int pos{1};
        for (auto const &arg : ref.args) {
          if (arg.index() != iter->second->signature[pos++]) {
            match = false;
            break;
          }
        }
        if (match) {
          return {iter->second};
        }
      }
    }
    return std::nullopt;
  }

private:
  std::multimap<std::string, const RteFunction *> functions;
};

// TODO: Find a better place to to host/Expr type conversions
std::optional<SomeHostType> ConvertToHostTypeIfConstant(
    const Expr<SomeType> &expr) {
  return std::visit(
      [](const auto &x) -> std::optional<SomeHostType> {
        using R = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<R, Expr<SomeReal>> ||
            std::is_same_v<R, Expr<SomeInteger>>) {
          // TODO: complex and character and logical
          return std::visit(
              [](const auto &y) -> std::optional<SomeHostType> {
                if (auto scalar{GetScalarConstantValue(y)}) {
                  return scalar->ConvertToHostType();
                }
                return std::nullopt;
              },
              x.u);
        } else {
          return std::nullopt;
        }
      },
      expr.u);
}

template<typename W, int P, bool IM>
using Real = typename Fortran::evaluate::value::Real<W, P, IM>;

template<typename T>
static Constant<T> ConvertHostTypeToIntrinsicType(HostType<T> value) {
  static_assert(!std::is_same_v<T, UnknowHostType>);
  static_assert(IsSpecificIntrinsicType<T>);
  if constexpr (T::category == TypeCategory::Real) {
    using WORD = typename Scalar<T>::Word;
    WORD *word = reinterpret_cast<WORD *>(&value);
    return Constant<T>{Scalar<T>{*word}};
  } else if constexpr (T::category == TypeCategory::Integer) {
    Scalar<T> *word = reinterpret_cast<Scalar<T> *>(&value);
    return Constant<T>{Scalar<T>{*word}};
  } else {
    return Constant<T>{};
  }
}

template<typename T>
std::optional<Constant<T>> ConvertFromHostType(const HostType<T> &value) {
  if constexpr (common::HasMember<T, RealTypes> ||
      common::HasMember<T, IntegerTypes>) {
    return {ConvertHostTypeToIntrinsicType<T>(value)};
  }
  return std::nullopt;
}
}
#endif  // FORTRAN_EVALUATE_RTE_BINDINGS_H_
