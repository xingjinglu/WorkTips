#include <vector>


template<typename V1, typename V2>
class vector_sum{

  public:
    vector_sum(const V1 &v1, const V2 &v2):v1_(v1), v2_(v2){}
    using value_type = std::common_type<typename V1::value_type,
          typename V2::value_type >;

    value_type operator[](int i){
      return v1_[i] + v2_[i];
    }

  private:
    const V1 & v1_;
    const V2 & v2_;
};


template <typename V1, typename V2>
inline vector_sum<V1, V2> operator+(const V1 &x, const V2 &y)
{
  return {x, y};
}
