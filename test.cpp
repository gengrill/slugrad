#include <ranges>
#include <vector>
#include <random>
#include <memory>
#include <iostream>
#include <functional>
#include <unordered_set>

using std::unordered_set;
using std::vector;
using std::string;
using std::function;
using std::shared_ptr;
using std::make_shared;

// class Value;
// template <class T>
// inline void hash_combine(std::size_t & seed, const T & v) {
//     std::hash<T> hasher; seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
// }
// struct ValueHash { size_t operator()(const Value& v) const; };
// inline size_t ValueHash::operator()(const Value& v) const {
//     std::size_t seed = 0;
//     hash_combine(seed, v.data); // this hashes the address of data and grad.. do we want that?
//     hash_combine(seed, v.grad);
//     return seed;
// }

struct Value {
  shared_ptr<float> data;
  shared_ptr<float> grad;
  function<void(void)> _backward;
  unordered_set<shared_ptr<Value>> _prev;
  string op;
  Value() : data(make_shared<float>(0.0f)), grad(make_shared<float>(0.0f)), _backward([](){}) {}
  //Value(const Value& v) : data(v.data), grad(v.grad), _backward(v._backward), _prev(v._prev), op(v.op) {}
  Value(const Value& v) : data(v.data), grad(v.grad), _backward(v._backward), _prev(v._prev), op(v.op) {}
  Value(float data) : data(make_shared<float>(data)), grad(make_shared<float>(0.0f)), _backward([](){}), op("") {}
  Value(float data, string op) : data(make_shared<float>(data)), grad(make_shared<float>(0.0f)), _backward([](){}), op(op) {}
  string print() const {
    return "(data=" + std::to_string(*(this->data)) + " , grad=" + std::to_string(*(this->grad)) + ", op='" + this->op + "')";
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
  // Value operator+(const float other) { return this + Value(other); }
  // bool operator==(const Value& v) const { return this.data.get() == v.data.get(); }
};

Value operator+(const Value& v1, const Value& v2) {
  Value out(*(v1.data) + *(v2.data), "+");
  out._prev.insert(make_shared<Value>(v1));
  out._prev.insert(make_shared<Value>(v2));
  out._backward = [&](void) {
    std::cout << "_backward" << std::endl;
    std::cout << "v1: " << v1.print() << std::endl;
    std::cout << "v2: " << v2.print() << std::endl;
    *(v1.grad) = *(v1.grad) + *(out.grad);
    *(v2.grad) = *(v2.grad) + *(out.grad);
    std::cout << "v1: " << v1.print() << std::endl;
    std::cout << "v2: " << v2.print() << std::endl;
  };
  return out;
}

Value operator*(const Value& v1, const Value& v2) {
  Value out(*v1.data * *v2.data, "*");
  std::cout << "out: " << out.print() << std::endl;
  out._prev.insert(make_shared<Value>(v1));
  out._prev.insert(make_shared<Value>(v2));
  out._backward = [&](void) {
    std::cout << "_backward" << std::endl;
    std::cout << "v1: " << v1.print() << std::endl;
    std::cout << "v2: " << v2.print() << std::endl;
    *(v1.grad) = *(v1.grad) + *(v2.data) * *(out.grad);
    *(v2.grad) = *(v2.grad) + *(v1.data) * *(out.grad);
    std::cout << "v1: " << v1.print() << std::endl;
    std::cout << "v2: " << v2.print() << std::endl;
  };
  return out;
}

int main(void) {
  Value a(13);
  Value b(-42.3);
  //  Value c = a + b;
  Value d = a * b;
  std::cout << "a: " << a.print() << std::endl;
  std::cout << "b: " << b.print() << std::endl;
  //  std::cout << "c: " << c.print() << std::endl;
  std::cout << "d: " << d.print() << std::endl;
  //  c.backward();
  std::cout << "a: " << a.print() << std::endl;
  std::cout << "b: " << b.print() << std::endl;
  //  std::cout << "c: " << c.print() << std::endl;
  std::cout << "d: " << d.print() << std::endl;
  d.backward();
  std::cout << "a: " << a.print() << std::endl;
  std::cout << "b: " << b.print() << std::endl;
  //  std::cout << "c: " << c.print() << std::endl;
  std::cout << "d: " << d.print() << std::endl;
  return 0;
}
