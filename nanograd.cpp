#include <cmath>
#include <ranges>
#include <vector>
#include <random>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <unordered_set>

using std::unordered_set;
using std::vector;
using std::string;
using std::mt19937;
using std::function;
using std::shared_ptr;
using std::make_shared;
using std::random_device;
using std::uniform_real_distribution;

namespace nanograd {

  float randn(int low, int high) {
    random_device dev;
    mt19937 rng(dev());
    uniform_real_distribution<float> prng(low, high);
    return prng(rng);
  }

  // Main data class, provides autograd functionality for scalars
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
    string print() const {
      return "Value(data=" + std::to_string(*(this->data)) + " , grad=" + std::to_string(*(this->grad)) + ", op='" + this->op + "')";
    }
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
      std::cout << "backward(): calling build_topo" << std::endl;
      build_topo(this);
      *grad = 1.0f;
      std::cout << print() << std::endl;
      std::cout << "backward(): calling _backward on topo" << std::endl;
      for (auto it=topo.rbegin(); it != topo.rend(); ++it)
        (*it)->_backward();
    }
    friend Value operator+(const Value& v1, const Value& v2);
    friend Value operator*(const Value& v1, const Value& v2);
    template<typename T> friend Value pow(const Value& base, T exponent);
    friend Value relu(const Value& v);
    Value operator+(const float other) { return *this + Value(other); }
  };

  Value operator+(const Value& v1, const Value& v2) {
    Value out(*(v1.data) + *(v2.data), "+");
    shared_ptr<Value> _v1 = make_shared<Value>(v1);
    shared_ptr<Value> _v2 = make_shared<Value>(v2);
    shared_ptr<Value> _out = make_shared<Value>(out);
    out._prev.insert(_v1);
    out._prev.insert(_v2);
    out._backward = [_v1, _v2, _out](void) {
      std::cout << "_backward +" << std::endl;
      std::cout << "v1: " << _v1->print() << std::endl;
      std::cout << "v2: " << _v2->print() << std::endl;
      *(_v1->grad) = *(_v1->grad) + *(_out->grad);
      *(_v2->grad) = *(_v2->grad) + *(_out->grad);
      std::cout << "v1: " << _v1->print() << std::endl;
      std::cout << "v2: " << _v2->print() << std::endl;
    };
    return out;
  }

  Value operator*(const Value& v1, const Value& v2) {
    Value out(*v1.data * *v2.data, "*");
    shared_ptr<Value> _v1 = make_shared<Value>(v1);
    shared_ptr<Value> _v2 = make_shared<Value>(v2);
    shared_ptr<Value> _out = make_shared<Value>(out);
    std::cout << "out: " << out.print() << std::endl;
    out._prev.insert(make_shared<Value>(v1));
    out._prev.insert(make_shared<Value>(v2));
    out._backward = [_v1, _v2, _out](void) {
      std::cout << "_backward *" << std::endl;
      std::cout << "v1: " << _v1->print() << std::endl;
      std::cout << "v2: " << _v2->print() << std::endl;
      *(_v1->grad) = *(_v1->grad) + *(_v2->data) * *(_out->grad);
      *(_v2->grad) = *(_v2->grad) + *(_v1->data) * *(_out->grad);
      std::cout << "v1: " << _v1->print() << std::endl;
      std::cout << "v2: " << _v2->print() << std::endl;
    };
    return out;
  }

  template<typename T>
  Value pow(const Value& base, const T exponent) {
    if constexpr (std::is_same_v<std::decay_t<const T>, int> || std::is_same_v<std::decay_t<const T>, float>) {
      Value out(std::pow((float)*base.data, (float)exponent), "pow("+std::to_string(exponent)+")");
      //shared_ptr<Value> _out = make_shared<Value>(out);
      out._prev.insert(make_shared<Value>(base));
      out._backward = [&base, exponent, &out](void) {
        std::cout << "_backward pow" << std::endl;
        std::cout << "base: " << base.print() << std::endl;
        std::cout << "exponent: " << std::to_string(exponent) << std::endl;
        *(base.grad) = *(base.grad) + exponent * (std::pow((float)*base.data, (float)exponent-1)) * (*out.grad);
        std::cout << "base: " << base.print() << std::endl;
        std::cout << "exponent: " << exponent << std::endl;
      };
      return out;
    }
    throw std::invalid_argument("Value::pow: received unknown type " + string(typeid(T).name()) + " for exponent (only int and floats are allowed).");
  }

  Value relu(const Value& v) {
    std::cout << "called ReLU on " << v.print() << std::endl;
    Value out(0 < *v.data ? *v.data : 0, "ReLU");
    shared_ptr<Value> _out = make_shared<Value>(out);
    shared_ptr<Value> _v = make_shared<Value>(v);
    out._prev.insert(_v);
    out._backward = [_v, _out](void) {
      std::cout << "_backward relu" << std::endl;
      *(_v->grad) = *(_v->grad) + (int(0 < *(_v->data)) * (*(_out->grad)));
    };
    std::cout << "calculated output value as " << out.print() << std::endl;
    return out;
  }

  class Module {
  public:
    void zero_grad() {
      for (auto p : parameters())
        *(p->grad) = 0;
    }
    virtual vector<shared_ptr<Value>> parameters() = 0; // TODO return iterator instead of vector?
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
      std::cout << "Called Neuron with input " << x.print() << std::endl;
      std::cout << "w.size() = " << w.size() << std::endl;
      std::cout << "*w[0] = " << w[0]->print() << std::endl;
      Value act = *w[0] * x + b;
      std::cout << "calculated Neuron activation as " << act.print() << std::endl;
      return nonlin ? nanograd::relu(act) : act;
    }
    Value operator() (const vector<shared_ptr<Value>> x) {
      std::cout << "Called Neuron with " << x.size() << " inputs." << std::endl;
      vector<shared_ptr<Value>> tmp;
      for (int i=0; i<w.size(); ++i)
        tmp.push_back(make_shared<Value>((*x[i]) * (*w[i])));
      //std::transform(x.begin(), x.end(), w.begin(), std::back_inserter(tmp), [] (const auto& xi, const auto& wi) {
      //  return (*xi) * (*wi);
      //});
      Value act;
      for (int i=0; i<tmp.size(); ++i)
        act = act + (*tmp[i]);
      //Value act(std::accumulate(tmp.begin(), tmp.end(), 0) + b);
      return nonlin ? nanograd::relu(act) : act;
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
      std::cout << "Called Layer with input " << x.print() << std::endl;
      std::cout << "neurons.size() = " << neurons.size() << std::endl;
      vector<shared_ptr<Value>> out;
      for (auto n : neurons) {
        Value act = (*n)(x);
        out.push_back(make_shared<Value>(act));
      }
      return out;
    }
    vector<shared_ptr<Value>> operator() (const vector<shared_ptr<Value>> x) {
      std::cout << "Called Layer with " << x.size() << " inputs." << std::endl;
      std::cout << "neurons.size() = " << neurons.size() << std::endl;
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
      std::cout << "Called MLP with input " << x.print() << std::endl;
      std::cout << "layers.size() = " << layers.size() << std::endl;
      vector<shared_ptr<Value>> out;
      out.push_back(make_shared<Value>(x));
      for (auto l : layers)
        out = (*l)(out);
      return out;
    }
    vector<shared_ptr<Value>> operator() (const vector<shared_ptr<Value>> x) {
      std::cout << "Called MLP with " << x.size() << " inputs." << std::endl;
      std::cout << "layers.size() = " << layers.size() << std::endl;
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
  return 0;
}
