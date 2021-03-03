#ifndef UTILS_H
#define UTILS_H

#include "rule.h"

int mine_rules(char** features, rule_t *samples, int nfeatures, int nsamples, 
                int max_card, double min_support, rule_t **rules_out, int verbose, int pre_mine);

int minority(rule_t* rules, int nrules, rule_t* labels, int nsamples, rule_t* minority_out, int verbose, int* minority_count, char* loss_type_str, double w);

#endif
