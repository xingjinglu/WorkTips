#include <iostream>
#include <string>

using namespace std;

int main() {
    string sequence;
    cin >> sequence;

    int boy_count = 0;
    int girl_count = 0;

    // 遍历字符串，寻找"boy"和"girl"单词
    for (size_t i = 0; i < sequence.length(); ++i) {
        // 检查是否有"boy"单词
        if ((i + 2< sequence.length()) && sequence.substr(i, 3) == "boy") {
            boy_count++;
            // 如果找到"boy"，跳过"b"和"o"，从"y"后面继续查找
            i += 2;
        }
        else if ((i + 1< sequence.length()) && sequence.substr(i, 2) == "bo") {
          boy_count++;
          // 如果找到"boy"，跳过"b"和"o"，从"y"后面继续查找
          i += 1;
        }
        else if ((i + 1< sequence.length()) && sequence.substr(i, 2) == "oy") {
          boy_count++;
          // 如果找到"boy"，跳过"b"和"o"，从"y"后面继续查找
          i += 1;
        }
        else if ( sequence.substr(i, 1) == "b") {
          boy_count++;
        }
        else if ( sequence.substr(i, 1) == "o") {
          boy_count++;
        }
        else if ( sequence.substr(i, 1) == "y") {
          boy_count++;
        }

        // 检查是否有"girl"单词
        else if (i + 3 < sequence.length() && sequence.substr(i, 4) == "girl") {
            girl_count++;
            // 如果找到"girl"，跳过"g"、"i"、"r"，从"l"后面继续查找
            i += 3;
        }
        else if (i + 2 < sequence.length() && sequence.substr(i, 3) == "gir") {
          girl_count++;
          // 如果找到"girl"，跳过"g"、"i"、"r"，从"l"后面继续查找
          i += 2;
        }
        else if (i + 2 < sequence.length() && sequence.substr(i, 3) == "irl") {
          girl_count++;
          // 如果找到"girl"，跳过"g"、"i"、"r"，从"l"后面继续查找
          i += 2;
        }
        else if (i + 1 < sequence.length() && sequence.substr(i, 2) == "gi") {
          girl_count++;
          // 如果找到"girl"，跳过"g"、"i"、"r"，从"l"后面继续查找
          i += 1;
        }
        else if (i + 1 < sequence.length() && sequence.substr(i, 2) == "ir") {
          girl_count++;
          // 如果找到"girl"，跳过"g"、"i"、"r"，从"l"后面继续查找
          i += 1;
        }
        else if (i + 1 < sequence.length() && sequence.substr(i, 2) == "rl") {
          girl_count++;
          // 如果找到"girl"，跳过"g"、"i"、"r"，从"l"后面继续查找
          i += 1;
       }
        else if (sequence.substr(i, 1) == "g" || sequence.substr(i, 1) == "i" || sequence.substr(i, 1) == "r" ||
          sequence.substr(i, 1) == "l" ) {
          girl_count++;
          // 如果找到"girl"，跳过"g"、"i"、"r"，从"l"后面继续查找
       }







    }

    // 输出结果
    cout << boy_count << endl;
    cout << girl_count << endl;

    return 0;
}
