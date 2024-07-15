#include "cosmo.hpp"
#include "logging.hpp"
#include "params.hpp"
#include "timestepper.hpp"

int main() {
    LOG_MINIMAL("git hash = %s", TOSTRING(GIT_HASH));
    LOG_MINIMAL("git modified = %s", TOSTRING(GIT_MODIFIED));

    Params params("test.params");
    Timestepper ts(params);
    Cosmo cosmo(params);

    LOG_MINIMAL("done!");
    return 0;
}