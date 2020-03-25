/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <stdexcept>

#include "flashlight/common/Logging.h"
#include "flashlight/nn/modules/Module.h"

#include "flashlight/nn/Init.h"

namespace fl {

Module::Module() = default;

Module::Module(const std::vector<std::pair<std::string, Variable>>& params) : params_(params.begin(), params.end()) {}
Module::Module(const std::vector<Variable>& params) {
  for (auto const& p : params) {
    params_.push_back({std::to_string(params_.size()), p});
  }
}

Variable Module::param(int position) const {
  return paramNamed(position).second;
}

std::pair<std::string, Variable> Module::paramNamed(int position) const {
  if (!(position >= 0 && position < params_.size())) {
    throw std::out_of_range("Module param index out of range");
  }
  return params_[position];
  
}

void Module::setParams(const Variable& var, int position) {
  if (!(position >= 0 && position < params_.size())) {
    throw std::out_of_range("Module param index out of range");
  }
  params_[position].second = var;
}

void Module::train() {
  train_ = true;
  for (auto& param : params_) {
    param.second.setCalcGrad(true);
  }
}

void Module::zeroGrad() {
  for (auto& param : params_) {
    param.second.zeroGrad();
  }
}

void Module::eval() {
  train_ = false;
  for (auto& param : params_) {
    param.second.setCalcGrad(false);
  }
}

std::vector<Variable> Module::params() const {
  std::vector<Variable> out(params_.size());
  std::transform(params_.begin(), params_.begin(), std::back_inserter(out), [](std::pair<std::string, Variable> p) { return p.second; });
  return out;
}

std::vector<Variable> Module::operator()(const std::vector<Variable>& input) {
  return this->forward(input);
}

void Module::loadStateDict(StateDict const& sd) {
  // Mapping of parameters - allows to ensure parameters uniqueness
  std::unordered_map<std::string, size_t> myParams;
  for (auto i = 0U; i < params_.size(); ++i) {
    auto const& p = params_[i];
    if (myParams.find(p.first) != myParams.end()) {
      throw std::runtime_error("Duplicate parameter with name " + p.first + " (parameters indices " + std::to_string(myParams[p.first]) + " and " + std::to_string(i) + ") in loadStateDict for module " + prettyString());
    }
    myParams[p.first] = i;
    if (sd.find(p.first) == sd.end()) {
      VLOG(1) << "Parameter " << p.first << " not in state dict";
    }
  }

  for (auto const& p: sd) {
    auto it = myParams.find(p.first);
    if (it == myParams.end()) {
      VLOG(1) << "Parameter " << p.first << " in state dict but not in current Module - ignored";
      continue;
    }
    auto currentParam = params_[it->second].second;
    if (currentParam.dims() != p.second.dims()) {
      throw std::runtime_error("Loading parameter with name " + p.first + ": size mismatch in loadStateDict for module" + prettyString());
    }
    setParams(p.second, it->second);
  }
}

StateDict Module::stateDict() const {
  std::unordered_map<std::string, Variable> myStateDict;
  for (auto i = 0U; i < params_.size(); ++i) {
    auto const& p = params_[i];
    if (myStateDict.find(p.first) != myStateDict.end()) {
      throw std::runtime_error("Duplicate parameter with name " + p.first + " in stateDict for module " + prettyString());
    }
    myStateDict[p.first] = p.second;
  }
  return myStateDict;
}

UnaryModule::UnaryModule() = default;

UnaryModule::UnaryModule(const std::vector<Variable>& params)
    : Module(params) {}

std::vector<Variable> UnaryModule::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() != 1) {
    throw std::invalid_argument("UnaryModule expects only one input");
  }
  return {forward(inputs[0])};
}

Variable UnaryModule::operator()(const Variable& input) {
  return this->forward(input);
}

BinaryModule::BinaryModule() = default;

BinaryModule::BinaryModule(const std::vector<Variable>& params)
    : Module(params) {}

std::vector<Variable> BinaryModule::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() != 2) {
    throw std::invalid_argument("BinaryModule expects two inputs");
  }
  return {forward(inputs[0], inputs[1])};
}

Variable BinaryModule::operator()(
    const Variable& input1,
    const Variable& input2) {
  return this->forward(input1, input2);
}

} // namespace fl
