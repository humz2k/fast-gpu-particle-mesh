#include "logging.hpp"
#include "params.hpp"

int main(){
    LOG_MINIMAL("git hash = %s",TOSTRING(GIT_HASH));
    LOG_MINIMAL("git modified = %s",TOSTRING(GIT_MODIFIED));

    Params("test.params");

    return 0;
}