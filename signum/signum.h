void signum_init();
int signum_samplerate();

/*------------------------------------------------------------------------
 * Core
 *-----------------------------------------------------------------------*/
#include "constants.h"
#include "unit.h"
#include "graph.h"
#include "buffer.h"

/*------------------------------------------------------------------------
 * Operators
 *-----------------------------------------------------------------------*/
#include "op/multiply.h"

/*------------------------------------------------------------------------
 * I/O
 *-----------------------------------------------------------------------*/
#include "io/output.h"

/*------------------------------------------------------------------------
 * Generators
 *-----------------------------------------------------------------------*/
#include "gen/constant.h"
#include "gen/sine.h"
#include "gen/square.h"
#include "gen/sampler.h"
#include "gen/granulator.h"

/*------------------------------------------------------------------------
 * Random processes
 *-----------------------------------------------------------------------*/
#include "rnd/noise.h"
#include "rnd/dust.h"

/*------------------------------------------------------------------------
 * Envelopes
 *-----------------------------------------------------------------------*/
#include "env/env.h"

/*------------------------------------------------------------------------
 * Effects
 *-----------------------------------------------------------------------*/
#include "fx/delay.h"
#include "fx/resample.h"

