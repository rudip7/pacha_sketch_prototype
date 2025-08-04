import math
import copy

class OmniSketch(Synopsis):
    def __init__(self, has_predicates=None, ram=None):
        super().__init__()
        self.setting = "OmniSketch"
        self.parameters = [Main.depth, Main.width, Main.max_size, Main.b]
        self.has_predicate = copy.deepcopy(has_predicates) if has_predicates is not None else None
        self.max_bits = [0] * Main.num_attributes
        self.die_hash_functions = []
        self.init_die_hashes()
        self.cmsketches = [None] * Main.num_attributes
        self.cmsketches_range = [[None for _ in range(Main.dyadic_range_bits + 1)] for _ in range(Main.num_attributes)]
        self.ram = ram
        if has_predicates is not None:
            if not Main.range_queries:
                for i in range(Main.num_attributes):
                    if has_predicates[i]:
                        self.cmsketches[i] = CountMin(self.die_hash_functions, i)
                    else:
                        self.cmsketches[i] = None
            else:
                for i in range(Main.num_attributes):
                    if has_predicates[i]:
                        for j in range(Main.dyadic_range_bits, -1, -1):
                            self.cmsketches_range[i][Main.dyadic_range_bits - j] = CountMinDyad(i, j)
                    else:
                        self.cmsketches[i] = None
        else:
            for i in range(Main.num_attributes):
                self.cmsketches[i] = CountMin(self.die_hash_functions, i)

        self.kmin_tmp = Kmin()
        self.min_cm_row = 0
        self.within_constraint = 0
        self.outside_constraint = 0
        self.ns = [[0 for _ in range(Main.depth)] for _ in range(Main.num_attributes)]
        self.bs = [[0 for _ in range(Main.depth)] for _ in range(Main.num_attributes)]

    def init_die_hashes(self):
        for i in range(Main.depth):
            self.die_hash_functions.append(DieHash(Main.max_level, 1))

    def add(self, record):
        id_ = record.get_id()
        hx = self.kmin_tmp.hash(id_)
        record_array = record.get_record()
        if not Main.range_queries:
            for i in range(Main.num_attributes):
                if self.has_predicate[i]:
                    self.cmsketches[i].add(id_, record_array[i], hx)
        else:
            for i in range(Main.num_attributes):
                if self.has_predicate[i]:
                    ranges = self.wrapper_init_log_ranges(record_array[i])
                    for j in range(len(ranges[2])):
                        self.cmsketches_range[i][j].add(ranges[1][j], ranges[2][j], hx)

    def wrapper_init_log_ranges(self, l):
        if l < 0:
            print("Error: l < 0")
            exit(1)
        ranges = self.get_log_ranges(l + 1)
        for i in range(len(ranges[2])):
            ranges[1][i] -= 1
            ranges[2][i] -= 1
        return ranges

    @staticmethod
    def get_log_ranges(input_key):
        coeff = [0] * Main.dyadic_range_bits
        lower_bound = [0] * Main.dyadic_range_bits
        upper_bound = [0] * Main.dyadic_range_bits

        half_point = int(math.pow(2, Main.dyadic_range_bits - 1))
        if input_key < half_point:
            coeff[0] = 0
            lower_bound[0] = 1
            upper_bound[0] = half_point
        else:
            coeff[0] = 1
            lower_bound[0] = half_point
            upper_bound[0] = half_point * 2

        pow_ = half_point
        for i in range(1, Main.dyadic_range_bits - 1):
            prev_coeff = coeff[i - 1]
            new_coeff_lower = prev_coeff * 2
            pow_ //= 2
            if (new_coeff_lower + 1) * pow_ < input_key:
                new_coeff_lower += 1
                coeff[i] = new_coeff_lower
            else:
                coeff[i] = new_coeff_lower
            lower_bound[i] = coeff[i] * pow_ + 1
            upper_bound[i] = (coeff[i] + 1) * pow_
        lower_bound[Main.dyadic_range_bits - 1] = input_key
        upper_bound[Main.dyadic_range_bits - 1] = input_key
        coeff[Main.dyadic_range_bits - 1] = input_key

        return [coeff, lower_bound, upper_bound]

    def query(self, q):
        s_cap = 0
        n_max = 0
        samples = [None] * (len(q.pred_attrs) * Main.depth)
        if Main.use_ds:
            raise RuntimeError("Not implemented")
        else:
            for i in range(len(q.pred_attrs)):
                temp = self.cmsketches[q.pred_attrs[i]].query(q.get_record()[q.pred_attrs[i]])
                if Main.depth >= 0:
                    samples[i * Main.depth:(i + 1) * Main.depth] = temp
            n_max = self.get_nmax(samples)
            s_cap = self.get_alt_est_kmv(samples)
        constraint = 3 * math.log((4 * len(q.pred_attrs) * Main.depth * math.sqrt(Main.max_size)) / Main.delta) / (Main.eps * Main.eps)
        q.case2_estimate = math.ceil(s_cap * n_max / Main.max_size)
        b_constraint = 2 * n_max * math.log((4 * len(q.pred_attrs) * Main.depth * math.sqrt(Main.max_size)) / Main.delta) / (q.exact_answer * Main.eps * Main.eps)
        if Main.max_size >= b_constraint:
            q.ratio_condition = True
        q.intersect_size = s_cap
        if s_cap < constraint:
            return int(math.ceil(2 * n_max * math.log((4 * len(q.pred_attrs) * Main.depth * math.sqrt(Main.max_size)) / Main.delta) / (Main.max_size * Main.eps * Main.eps)))
        else:
            q.thrm33_case2 = True
            return int(math.ceil(s_cap * n_max / Main.max_size))

    def get_nmax(self, samples):
        n_max = 0
        for sample in samples:
            if sample.n > n_max:
                n_max = sample.n
        return n_max

    def get_estimate_kmv(self, samples):
        c = 0
        iter_ = iter(samples[0].sketch)
        for i in iter_:
            found = True
            for j in range(1, len(samples)):
                if i not in samples[j].sketch:
                    found = False
                    break
            if found:
                c += 1
        return c

    def get_alt_est_kmv(self, samples):
        num_joins = len(samples)
        c = 0
        iter_ = iter(samples[0].sketch)
        while iter_ is not None:
            try:
                i = next(iter_)
            except StopIteration:
                break
            found = True
            for j in range(1, num_joins):
                other_element = samples[j].sketch.ceiling(i)
                if other_element is None:
                    found = False
                    iter_ = None
                    break
                elif other_element == i:
                    continue
                else:
                    iter_ = iter(samples[0].sketch.tail_set(other_element))
                    found = False
                    break
            if found:
                c += 1
        return c

    def get_alt_est_kmv_treeset(self, samples):
        num_joins = len(samples)
        c = 0
        iter_ = iter(samples[0])
        while iter_ is not None:
            try:
                i = next(iter_)
            except StopIteration:
                break
            found = True
            for j in range(1, num_joins):
                other_element = samples[j].ceiling(i)
                if other_element is None:
                    found = False
                    iter_ = None
                    break
                elif other_element == i:
                    continue
                else:
                    iter_ = iter(samples[0].tail_set(other_element))
                    found = False
                    break
            if found:
                c += 1
        return c

    def get_retain_all(self, samples):
        intersect = set(samples[0].sketch)
        for i in range(1, len(samples)):
            intersect &= set(samples[i].sketch)
        return len(intersect)

    def get_intersected_estimate_ds(self, query, samples):
        intersect_sample = None
        for j in range(Main.depth):
            for i in range(len(query.pred_attrs)):
                if i == 0 and j == 0:
                    intersect_sample = DistinctSample(samples[i][j])
                else:
                    other = samples[i][j]
                    intersect_sample = intersect_sample.intersect(other)
        assert intersect_sample is not None
        est_sample = self.get_size_and_level(intersect_sample)
        estimate = math.pow(2, est_sample.sample_level + 1) * est_sample.intersect_size
        query.intersect_size = est_sample.intersect_size
        query.sample_level = est_sample.sample_level
        return estimate

    def get_nmax_ns(self):
        n_max = 0
        for i in range(Main.num_attributes):
            for j in range(Main.depth):
                if self.ns[i][j] > n_max:
                    n_max = self.ns[i][j]
        return n_max

    def range_query(self, q):
        s_cap = 0
        n_max = 0
        b_virtual = 0
        samples = [None] * (len(q.pred_attrs) * Main.depth)
        for i in range(Main.num_attributes):
            for j in range(Main.depth):
                self.ns[i][j] = 0
        for i in range(len(q.pred_attrs)):
            attr = q.pred_attrs[i]
            ranges_list = self.wrap_log_ranges(q.lower[attr], q.upper[attr])
            set_ = [None] * Main.depth
            b_virtual = len(ranges_list) * Main.max_size
            for j in range(len(ranges_list)):
                cm = self.cmsketches_range[q.pred_attrs[i]][self.get_index_of_range(ranges_list[j])]
                range_set = cm.range_query(ranges_list[j][0], ranges_list[j][1], self.ns[attr])
                for d in range(Main.depth):
                    if set_[d] is None:
                        set_[d] = range_set[d]
                    else:
                        set_[d].update(range_set[d])
            if Main.depth >= 0:
                samples[i * Main.depth:(i + 1) * Main.depth] = set_
        n_max = self.get_nmax_ns()
        s_cap = self.get_alt_est_kmv_treeset(samples)
        constraint = 3 * math.log((4 * len(q.pred_attrs) * Main.depth * math.sqrt(b_virtual)) / Main.delta) / (Main.eps * Main.eps)
        q.case2_estimate = math.ceil(s_cap * n_max / b_virtual)
        b_constraint = 2 * n_max * math.log((4 * len(q.pred_attrs) * Main.depth * math.sqrt(b_virtual)) / Main.delta) / (q.exact_answer * Main.eps * Main.eps)
        if b_virtual >= b_constraint:
            q.ratio_condition = True
        q.intersect_size = s_cap
        if s_cap < constraint:
            return int(math.ceil(2 * n_max * math.log((4 * len(q.pred_attrs) * Main.depth * math.sqrt(b_virtual)) / Main.delta) / (b_virtual * Main.eps * Main.eps)))
        else:
            q.thrm33_case2 = True
            return int(math.ceil(s_cap * n_max / b_virtual))

    def wrap_log_ranges(self, low, up):
        if low < 0 or up < 0:
            print(f"Error because low or up < 0: low: {low} up: {up}")
            exit(1)
        temp = self.get_log_ranges_arr_list(low + 1, up + 1)
        for i in temp:
            i[0] -= 1
            i[1] -= 1
        return temp

    def get_index_of_range(self, longs):
        return int(math.log(longs[1] - longs[0] + 1) / math.log(2))

    @staticmethod
    def get_log_ranges_arr_list(start_inclusive, stop_inclusive):
        start_inclusive -= 1
        stop_inclusive -= 1
        init_diff = stop_inclusive - start_inclusive + 1
        result = []
        total_sum = 0
        pow_ = 1
        for j in range(Main.dyadic_range_bits):
            if start_inclusive + pow_ - 1 > stop_inclusive:
                break
            elif start_inclusive % (pow_ * 2) == 0 and start_inclusive + pow_ - 1 <= stop_inclusive:
                pass
            else:
                result.append([1 + start_inclusive, 1 + start_inclusive + pow_ - 1])
                total_sum += pow_
                start_inclusive += pow_
            pow_ *= 2

        pow_ = int(math.pow(2, Main.dyadic_range_bits))
        for j in range(Main.dyadic_range_bits, -1, -1):
            if start_inclusive % pow_ == 0 and start_inclusive + pow_ - 1 <= stop_inclusive:
                result.append([1 + start_inclusive, 1 + start_inclusive + pow_ - 1])
                total_sum += pow_
                start_inclusive += pow_
            pow_ //= 2

        if total_sum != init_diff:
            print("Error - no full coverage")
        return result

    def get_size_and_level(self, intersect_sample):
        while True:
            assert intersect_sample is not None
            if not (intersect_sample.sample.get(intersect_sample.sample_level) and len(intersect_sample.sample.get(intersect_sample.sample_level)) == 0):
                break
            intersect_sample.sample_level += 1
        if intersect_sample.sample.get(intersect_sample.sample_level):
            intersect_sample.set_intersect_size(len(intersect_sample.sample.get(intersect_sample.sample_level)))
        if Main.max_level < intersect_sample.sample_level:
            raise Exception("Max level surpassed")
        return intersect_sample

    def get_plain_estimate_ds(self, query, samples):
        estimate = float('inf')
        for j in range(Main.depth):
            intersect_sample = None
            for i in range(len(query.pred_attrs)):
                if i == 0:
                    intersect_sample = DistinctSample(samples[i][j])
                else:
                    other = samples[i][j]
                    intersect_sample = intersect_sample.intersect(other)
            assert intersect_sample is not None
            est_sample = self.get_size_and_level(intersect_sample)
            if math.pow(2, est_sample.sample_level + 1) * est_sample.intersect_size < estimate:
                self.min_cm_row = j
                estimate = math.pow(2, est_sample.sample_level + 1) * est_sample.intersect_size
                query.intersect_size = est_sample.intersect_size
                query.sample_level = est_sample.sample_level
        return estimate

    def reset(self):
        for i in range(Main.num_attributes):
            if self.has_predicate[i]:
                if Main.range_queries:
                    for j in range(Main.dyadic_range_bits + 1):
                        self.cmsketches_range[i][j].reset()
                else:
                    self.cmsketches[i].reset()
