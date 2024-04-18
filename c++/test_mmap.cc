#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <fstream>

#include "MmapUtils.h"

int test_file_size()
{
  int fd = open("example.txt", O_RDONLY);
  if (fd == -1) {
    std::cerr << "Failed to open file" << std::endl;
    return 1;
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    std::cerr << "Failed to get file size" << std::endl;
    return 1;
  }
  else
    std::cout<<"FileSize: " << sb.st_size <<std::endl;

  if (fstat(fd, &sb) == -1) {
    std::cerr << "Failed to get file size" << std::endl;
    return 1;
  }
  else
    std::cout<<"FileSize: " << sb.st_size <<std::endl;


  // 
  std::ifstream file("block_0.mnn", std::ios::binary|std::ios::ate);
  if(!file.is_open()){

    return -1;
  }

  std::streamsize size = file.tellg();

  // line number.
  int line_count = 0;
  std::string line;
  file.seekg(0, std::ios::beg);
  while(std::getline(file, line)){
    line_count++;
  }
  file.close();
  std::cout<<"line_count = " << line_count;
  std::cout<<"File size is : " << size - line_count << std::endl;


  return 0;
}


int test()
{
  int fd = open("example.txt", O_RDONLY);
  if (fd == -1) {
    std::cerr << "Failed to open file" << std::endl;
    return 1;
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    std::cerr << "Failed to get file size" << std::endl;
    return 1;
  }
  else
    std::cout<<"FileSize: " << sb.st_size <<std::endl;

  //
  char *file_in_memory = static_cast<char*>(mmap(NULL, sb.st_size, PROT_WRITE|PROT_READ, MAP_PRIVATE, fd, 0));
  if (file_in_memory == MAP_FAILED) {
    std::cerr << "Failed to mmap file" << std::endl;
    return 1;
  }

  for (size_t i = 0; i < sb.st_size; ++i) {
    std::cout << file_in_memory[i];
  }
  std::cout << std::endl;

  munmap(file_in_memory, sb.st_size);
  close(fd);

  return 0;
}

int main() {
  // test MmapStorage
  MmapStorage<int> m(16);
  int *ptr = m.get();
  for(int i = 0; i < 16; i++)
    ptr[i] = 16 + i;
  for(int i = 0; i < 16; i++)
    std::cout<<ptr[i]<<std::endl;
  m.release();
  if(m.get() == nullptr)
    std::cout<<"MmapStorage released " << std::endl;
  else
    std::cout<<"MmapStorage not released " << std::endl;


  // 
  test_file_size();

  // 
  int *m_ptr = static_cast<int*>(MmapAllocAlign(128));
  std::cout<<"m_ptr = " << m_ptr << std::endl;
  for(int i = 0; i < 2; i++)
    m_ptr[i] = i + 2;
  MmapFree(m_ptr);
  return 0;
}
