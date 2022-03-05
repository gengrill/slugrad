// ~/projects/llvm-13/build/bin/clang++ -std=c++20 nanograd.cpp -o nanograd
#include <cmath>
#include <tuple>
#include <ranges>
#include <vector>
#include <random>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <string_view>
#include <unordered_set>

using std::tuple;
using std::vector;
using std::string;
using std::mt19937;
using std::function;
using std::to_string;
using std::shared_ptr;
using std::make_shared;
using std::random_device;
using std::unordered_set;
using std::uniform_real_distribution;

namespace nanograd {

  static const bool LOGGING=false;
  static inline void log(string fmt, string lvl="DEBUG") {
    if constexpr (LOGGING) {
      std::cout<<"["+lvl+"]: "+fmt<<std::endl;
    }
  }

  float randn(int low, int high) {
    random_device dev;
    mt19937 rng(dev());
    uniform_real_distribution<float> prng(low, high);
    return prng(rng);
  }

  tuple<vector<float>,vector<float>> make_moons(size_t samples, float noise) {
    vector<float> X, Y;
    for (size_t i=0; i<samples; ++i) {
    }
    return std::make_tuple<>(X,Y);
  }

  /** Main data class, provides autograd functionality by wrapping single-precision scalars. **/
  struct Value {
    shared_ptr<float> data;
    shared_ptr<float> grad;
    function<void(void)> _backward;
    unordered_set<shared_ptr<Value>> _prev;
    string op;
    Value() : data(make_shared<float>(0.0f)), grad(make_shared<float>(0.0f)), _backward([](){}) {}
    Value(const Value& v) : data(v.data), grad(v.grad), _backward(v._backward), _prev(v._prev), op(v.op) {}
    Value(float data) : data(make_shared<float>(data)), grad(make_shared<float>(0.0f)), _backward([](){}), op("") {}
    Value(float data, string op) : data(make_shared<float>(data)), grad(make_shared<float>(0.0f)), _backward([](){}), op(op) {}
    string print() const { return "Value(data="+to_string(*(this->data))+", grad="+to_string(*(this->grad))+", op='"+this->op+"')"; }
    void backward() {
      vector<Value*> topo;
      unordered_set<Value*> visited;
      function<void(Value*)> build_topo = [&](Value *v) {
        if (visited.find(v) == visited.end()) {
          visited.insert(v);
          for (auto child : v->_prev)
            build_topo(child.get());
          topo.push_back(v);
        }
      };
      build_topo(this);
      *grad = 1.0f;
      for (auto it=topo.rbegin(); it != topo.rend(); ++it)
        (*it)->_backward();
    }
    friend Value operator+(const Value& v1, const Value& v2);
    friend Value operator-(const Value& v1, const Value& v2);
    friend Value operator*(const Value& v1, const Value& v2);
    friend Value operator/(const Value& v1, const Value& v2);
    template<typename T> friend Value pow(const Value& base, T exponent);
    friend Value relu(const Value& v);
    Value operator+(const float other) { return *this + Value(other); }
    Value operator*(const float other) { return *this * Value(other); }
  };

  Value operator+(const Value& v1, const Value& v2) {
    Value out(*(v1.data) + *(v2.data), "+");
    out._prev.insert(make_shared<Value>(v1));
    out._prev.insert(make_shared<Value>(v2));
    out._backward = [=](void) {
      log("before _backward +[v1="+v1.print()+", v2="+v2.print()+"]");
      *(v1.grad) = *(v1.grad) + *(out.grad);
      *(v2.grad) = *(v2.grad) + *(out.grad);
      log("after _backward +[v1="+v1.print()+", v2="+v2.print()+"]");
    };
    return out;
  }

  Value operator-(const Value& v1, const Value& v2) { return v1 + (-1.0 * v2); }

  Value operator*(const Value& v1, const Value& v2) {
    Value out(*v1.data * *v2.data, "*");
    log("out: " + out.print());
    out._prev.insert(make_shared<Value>(v1));
    out._prev.insert(make_shared<Value>(v2));
    out._backward = [=](void) {
      log("before _backward *[v1="+v1.print()+", v2="+v2.print());
      *(v1.grad) = *(v1.grad) + *(v2.data) * *(out.grad);
      *(v2.grad) = *(v2.grad) + *(v1.data) * *(out.grad);
      log("after _backward *[v1="+v1.print()+", v2="+v2.print());
    };
    return out;
  }

  Value operator/(const Value& v1, const Value& v2) { return v1 * pow(v2, -1); }

  template<typename T>
  Value pow(const Value& base, const T exponent) {
    if constexpr (std::is_same_v<std::decay_t<const T>, int> || std::is_same_v<std::decay_t<const T>, float>) {
      Value out(std::pow((float)*base.data, (float)exponent), "pow("+to_string(exponent)+")");
      out._prev.insert(make_shared<Value>(base));
      out._backward = [=](void) {
        log("before _backward pow[base="+base.print()+", exponent="+to_string(exponent));
        *(base.grad) = *(base.grad) + exponent * (std::pow((float)*base.data, (float)exponent-1)) * (*out.grad);
        log("after _backward pow[base="+base.print()+", exponent="+to_string(exponent));
      };
      return out;
    }
    throw std::invalid_argument("Value::pow: received unknown type " + string(typeid(T).name()) + " for exponent (only int and floats are allowed).");
  }

  Value relu(const Value& v) {
    Value out(0 < *v.data ? *v.data : 0, "ReLU");
    out._prev.insert(make_shared<Value>(v));
    out._backward = [=](void) {
      log("before _backward relu[v="+v.print());
      *(v.grad) = *(v.grad) + (int(0 < *(v.data)) * (*(out.grad)));
      log("after _backward relu[v="+v.print());
    };
    return out;
  }

  class Module {
  public:
    void zero_grad() {
      for (auto p : parameters())
        *(p->grad) = 0;
    }
    virtual vector<shared_ptr<Value>> parameters() = 0;
  };

  class Neuron : public Module {
    vector<shared_ptr<Value>> w;
    Value b;
    bool nonlin;
  public:
    Neuron(size_t nin, bool linear=true) : b(0.0f), nonlin(!linear) {
      for (int i=0; i<nin; ++i)
        w.push_back(make_shared<Value>(randn(-1, 1)));
    }
    string print() const {
      string out = (nonlin ? "Neuron" : "LinearNeuron");
      out += "(w=[";
      for (auto vp : w)
        out += vp->print() + ", ";
      out += "], b=" + b.print() + ")";
      return out;
    }
    vector<shared_ptr<Value>> parameters() {
      vector<shared_ptr<Value>> params(w);
      params.push_back(make_shared<Value>(b));
      return params;
    }
    Value operator() (const Value& x) {
      log("Called Neuron with input " + x.print());
      Value act = *w[0] * x + b;
      log("calculated Neuron activation as " + act.print());
      return nonlin ? nanograd::relu(act) : act;
    }
    Value operator() (const vector<shared_ptr<Value>> x) {
      log("Called Neuron with " + to_string(x.size()) + " inputs.");
      Value act;
      for (int i=0; i<w.size(); ++i)
        act = act + (*x[i]) * (*w[i]);
      act = nonlin ? nanograd::relu(act) : act;
      log("calculated Neuron activation as " + act.print());
      return act;
    }
  };

  class Layer : public Module {
    vector<shared_ptr<Neuron>> neurons;
  public:
    Layer(size_t nin, size_t nout, bool linear=true) {
      for (int i=0; i<nout; ++i) {
        Neuron in(nin, linear);
        neurons.push_back(make_shared<Neuron>(in));
      }
    }
    vector<shared_ptr<Value>> parameters() {
      vector<shared_ptr<Value>> params;
      for (auto n : neurons)
        for (auto p : n->parameters())
          params.push_back(p);
      return params;
    }
    vector<shared_ptr<Value>> operator() (const Value& x) {
      log("Called Layer with input " + x.print());
      log("neurons.size() = " + to_string(neurons.size()));
      vector<shared_ptr<Value>> out;
      for (auto n : neurons) {
        Value act = (*n)(x);
        out.push_back(make_shared<Value>(act));
      }
      return out;
    }
    vector<shared_ptr<Value>> operator() (const vector<shared_ptr<Value>> x) {
      log("Called Layer with " + to_string(x.size()) + " inputs.");
      log("neurons.size() = " + to_string(neurons.size()));
      vector<shared_ptr<Value>> out;
      for (auto n : neurons) {
        Value act = (*n)(x);
        out.push_back(make_shared<Value>(act));
      }
      return out;
    }
  };

  class MLP : public Module {
    vector<shared_ptr<Layer>> layers;
  public:
    MLP(size_t nin, vector<size_t> nouts) {
      Layer in(nin, nouts[0]);
      layers.push_back(make_shared<Layer>(in));
      for (int i=0; i<nouts.size()-2; ++i) {
        Layer hidden(nouts[i], nouts[i+1]);
        layers.push_back(make_shared<Layer>(hidden));
      }
      Layer out(nouts[nouts.size()-1], nouts[nouts.size()-1], false);
      layers.push_back(make_shared<Layer>(out));
    }
    vector<shared_ptr<Value>> parameters() {
      vector<shared_ptr<Value>> params;
      for (auto l : layers)
        for (auto p : l->parameters())
          params.push_back(p);
      return params;
    }
    vector<shared_ptr<Value>> operator() (const Value& x) {
      log("Called MLP with input " + x.print());
      log("layers.size() = " + to_string(layers.size()));
      vector<shared_ptr<Value>> out;
      out.push_back(make_shared<Value>(x));
      for (auto l : layers)
        out = (*l)(out);
      return out;
    }
    vector<shared_ptr<Value>> operator() (const vector<shared_ptr<Value>> x) {
      log("Called MLP with " + to_string(x.size()) + " inputs.");
      log("layers.size() = " + to_string(layers.size()));
      vector<shared_ptr<Value>> out(x);
      for (auto l : layers)
        out = (*l)(out);
      return out;
    }    
  };
}// namespace nanograd

int main(void) {
  nanograd::Value a(13);
  nanograd::Value b(-42.3);
  nanograd::Value c = a + b;
  nanograd::Value d = nanograd::pow(a, 2);
  std::cout << "a: " << a.print() << std::endl;
  std::cout << "b: " << b.print() << std::endl;
  std::cout << "c: " << c.print() << std::endl;
  std::cout << "d: " << d.print() << std::endl;
  c.backward();
  std::cout << "a: " << a.print() << std::endl;
  std::cout << "b: " << b.print() << std::endl;
  std::cout << "c: " << c.print() << std::endl;
  std::cout << "d: " << d.print() << std::endl;
  d.backward();
  std::cout << "a: " << a.print() << std::endl;
  std::cout << "b: " << b.print() << std::endl;
  std::cout << "c: " << c.print() << std::endl;
  std::cout << "d: " << d.print() << std::endl;
  nanograd::Neuron in(2);
  nanograd::Value x1(99.0f);
  nanograd::Value x2(1.22f);
  vector<shared_ptr<nanograd::Value>> x;
  x.push_back(make_shared<nanograd::Value>(x1));
  x.push_back(make_shared<nanograd::Value>(x2));
  nanograd::Value y1 = in(x);
  std::cout << "Got output from Neuron: " << y1.print() << std::endl;
  std::cout << in.print() << std::endl;
  y1.backward();
  std::cout << in.print() << std::endl;
  nanograd::Layer input(2, 2);
  vector<shared_ptr<nanograd::Value>> outputs = input(x);
  std::cout << "Got " << outputs.size() << " outputs from layer" << std::endl;
  for (auto o : outputs)
    o->backward();
  nanograd::MLP model(2, {16, 16, 1});
  vector<shared_ptr<nanograd::Value>> ys = model(x);
  std::cout << "Got prediction " << ys[0]->print() << " from MLP" << std::endl;
  ys[0]->backward();
  std::cout << "Printing model parameters:" << std::endl;
  for (auto p : model.parameters())
    if (*(p->grad) != 0)
      std::cout << p->print() << std::endl;
  return 0;
}
