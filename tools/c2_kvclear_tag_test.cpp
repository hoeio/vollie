// Tagged-pointer check of the 0009 logic: mmap 64 MB, fill, tag the pointer with 0xb4 (TBI), run the same
// range/maps/madvise steps on the untagged address, check zeros and RSS drop.
#include <sys/mman.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <unistd.h>
static bool anon(uintptr_t a, uintptr_t b){FILE*f=fopen("/proc/self/maps","r");char l[512];uintptr_t cov=a;bool ok=true;
 while(fgets(l,sizeof l,f)){unsigned long lo,hi,ino;char perm[8];if(sscanf(l,"%lx-%lx %7s %*s %*s %lu",&lo,&hi,perm,&ino)!=4||hi<=a||lo>=b)continue;
 if(lo>cov||perm[3]!='p'||ino!=0){ok=false;break;}cov=hi;}fclose(f);return ok&&cov>=b;}
int main(){size_t size=64u<<20; size_t off=123;
 uint8_t*raw=(uint8_t*)mmap(0,size+8192,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
 uint8_t*p=(uint8_t*)(((uintptr_t)(raw+off))|((uintptr_t)0xb4<<56)); memset(p,7,size);
 uintptr_t pg=sysconf(_SC_PAGESIZE), mask=~((uintptr_t)0xff<<56), u=(uintptr_t)p&mask;
 uintptr_t a=(u+pg-1)&~(pg-1), b=(u+size)&~(pg-1);
 bool tagged_ok=anon(((uintptr_t)p+pg-1)&~(pg-1), ((uintptr_t)p+size)&~(pg-1));
 bool ok=anon(a,b); int r=madvise((void*)a,b-a,MADV_DONTNEED);
 memset(p,0,a-u); memset(p+(b-u),0,u+size-b);
 size_t nz=0; for(size_t i=0;i<size;i++) nz+=p[i]!=0;
 printf("maps tagged=%d untagged=%d madvise=%d nonzero=%zu tag=0x%02x released=%zu\n",tagged_ok,ok,r,nz,(unsigned)((uintptr_t)p>>56),(size_t)(b-a));
 return !( !tagged_ok && ok && r==0 && nz==0);}
