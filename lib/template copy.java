
import org.apache.commons.lang3.ArrayUtils;

import java.util.*;

public class OmniSketch extends Synopsis {
    CountMin[] CMSketches = new CountMin[Main.numAttributes];
    CountMinDyad[][] CMSketchesRange = new CountMinDyad[Main.numAttributes][Main.dyadicRangeBits + 1];
    ArrayList<DieHash> dieHashFunctions = new ArrayList<>();
    boolean[] hasPredicate;

    public OmniSketch(boolean[] hasPredicates) {
        super();
        setting = "OmniSketch";
        System.out.println("Epsilon: " + Main.eps);
        System.out.println("Delta: " + Main.delta);
        System.out.println("Depth: " + Main.depth);
        System.out.println("Width: " + Main.width);
        System.out.println("Max Sample Size: " + Main.maxSize);
        System.out.println("b: " + Main.b);
        this.parameters = new int[]{Main.depth, Main.width, Main.maxSize, Main.b};
        this.hasPredicate = ArrayUtils.clone(hasPredicates);
        this.maxBits = new int[Main.numAttributes];
        initDieHashes();
        if (!Main.rangeQueries) {
            for (int i = 0; i < Main.numAttributes; i++) {
                if (hasPredicates[i]) {
                    CMSketches[i] = new CountMin(dieHashFunctions, i);
                } else {
                    CMSketches[i] = null;
                }
            }
        } else {
            // Make dyadic intervals per attribute sketch
            for (int i = 0; i < Main.numAttributes; i++) {
                if (hasPredicates[i]) {
                    for (int j = Main.dyadicRangeBits; j >-1; j--) {
                        CMSketchesRange[i][Main.dyadicRangeBits - j] = new CountMinDyad(i, j);
                    }
                } else {
                    CMSketches[i] = null;
                }
            }
        }
    }

    public OmniSketch(boolean[] hasPredicates, long ram) {
        this.setting = "OmniSketch";
        this.ram = ram;
        System.out.println("Depth: " + Main.depth);
        System.out.println("Width: " + Main.width);
        System.out.println("Max Sample Size: " + Main.maxSize);
        System.out.println("b: " + Main.b);
        this.parameters = new int[]{Main.depth, Main.width, Main.maxSize, Main.b};
        this.hasPredicate = ArrayUtils.clone(hasPredicates);
        this.maxBits = new int[Main.numAttributes];
        initDieHashes();
        if (!Main.rangeQueries) {
            for (int i = 0; i < Main.numAttributes; i++) {
                if (hasPredicates[i]) {
                    CMSketches[i] = new CountMin(dieHashFunctions, i);
                } else {
                    CMSketches[i] = null;
                }
            }
        } else {
            // Make dyadic intervals per attribute sketch
            for (int i = 0; i < Main.numAttributes; i++) {
                if (hasPredicates[i]) {
                    for (int j = Main.dyadicRangeBits; j >-1; j--) {
                        CMSketchesRange[i][Main.dyadicRangeBits - j] = new CountMinDyad(i, j);
                    }
                } else {
                    CMSketches[i] = null;
                }
            }
        }
    }

    public OmniSketch() {
        initDieHashes();
        for (int i = 0; i < Main.numAttributes; i++) {
            CMSketches[i] = new CountMin(dieHashFunctions, i);
        }
    }

    public void initDieHashes() {
        for (int i = 0; i < Main.depth; i++) {
            dieHashFunctions.add(new DieHash(Main.maxLevel, 1));
        }
    }

    Kmin kminTmp = new Kmin();
    public void add(Record record) {
        int id = record.getId();
        long hx = kminTmp.hash(id);
        long[] recordArray = record.getRecord();

        if (!Main.rangeQueries) {
            for (int i = 0; i < Main.numAttributes; i++) {
                if (hasPredicate[i]) {
                    CMSketches[i].add(id, recordArray[i], hx);
                }
            }
        } else {
            // Compute all dyadic ranges it belongs to.
            // Insert in all those ranges.
            for (int i = 0; i < Main.numAttributes; i++) {
                if (hasPredicate[i]) {
                    long[][] ranges = wrapperInitLogRanges(recordArray[i]);
                    for (int j = 0; j < ranges[2].length; j++) {
                        CMSketchesRange[i][j].add(ranges[1][j], ranges[2][j], hx);
                    }
                }
            }
        }
    }

    private long[][] wrapperInitLogRanges(long l) {
        if (l < 0) {
            System.out.println("Error: l < 0");
            System.exit(1);
        }
        long[][] ranges = getLogRanges(l + 1);
        for (int i = 0; i < ranges[2].length; i++) {
            ranges[1][i] = ranges[1][i] - 1L ;//- (long) Math.pow(2, Main.dyadicRangeBits - 1);
            ranges[2][i] = ranges[2][i] - 1L ;//- (long) Math.pow(2, Main.dyadicRangeBits - 1);
        }
        return ranges;
    }

    public static long[][] getLogRanges(long inputKey) {
        long[] coeff      = new long[Main.dyadicRangeBits];
        long[] lowerBound = new long[Main.dyadicRangeBits]; // at pos i we store the x for ranges of size 2^(maxSize-i)
        long[] upperBound = new long[Main.dyadicRangeBits]; // at pos i we store the x for ranges of size 2^(maxSize-i)

        long halfPoint = (long) Math.pow(2,Main.dyadicRangeBits -1);
        if (inputKey < halfPoint) {
            coeff[0]=0;
            lowerBound[0] = 1; // inclusive, starting from 1
            upperBound[0] = halfPoint; // inclusive
        } else {
            coeff[0]=1;
            lowerBound[0] = halfPoint; //inclusive, starting from 1
            upperBound[0] = halfPoint*2; // inclusive
        }

        long pow = halfPoint;
        for (int i = 1; i<Main.dyadicRangeBits -1; i++) {
            long prevCoeff = coeff[i-1];
            long newCoeffLower = prevCoeff*2;
            pow/=2;
            if ((newCoeffLower + 1) *pow < inputKey) {
                newCoeffLower++;
                coeff[i] = newCoeffLower;
            } else {
                coeff[i] = newCoeffLower;
            }
            lowerBound[i] = coeff[i]*pow+1;
            upperBound[i] = (coeff[i]+1)*pow;
        }
        lowerBound[Main.dyadicRangeBits -1] = inputKey;
        upperBound[Main.dyadicRangeBits -1] = inputKey;
        coeff[Main.dyadicRangeBits -1] = inputKey;

        long[][] result = new long[3][];
        result[0] = coeff;
        result[1] = lowerBound;
        result[2] = upperBound;
        return result;
    }

    int minCMRow = 0;

    public int withinConstraint= 0;
    public int outsideConstraint = 0;
    public int query(Query q) {
        double S_cap = 0;
        int n_max = 0;
        Sample[] samples = new Kmin[q.predAttrs.size() * Main.depth];
        if (Main.useDS) {
           throw new RuntimeException("Not implemented");
        } else {
          for (int i = 0; i < q.predAttrs.size(); i++) {
              Sample[] temp = CMSketches[q.predAttrs.get(i)].query(q.getRecord()[q.predAttrs.get(i)]);
              if (Main.depth >= 0) System.arraycopy(temp, 0, samples, i * Main.depth, Main.depth);
          }
          n_max = getNmax((Kmin[]) samples);
          S_cap = getAltEstKMV((Kmin[]) samples);///Instead of getAltEstKMV
        }
        double constraint = 3 * Math.log((4 * q.predAttrs.size() * Main.depth * Math.sqrt(Main.maxSize))
                / Main.delta)/(Main.eps * Main.eps);
        q.case2Estimate = Math.ceil(S_cap * n_max / Main.maxSize);
        double BConstraint =  2 * n_max * Math.log((4 * q.predAttrs.size() * Main.depth *
                Math.sqrt(Main.maxSize)) / Main.delta)/(q.exactAnswer * Main.eps * Main.eps);
        if (Main.maxSize >= BConstraint) {
            q.ratioCondition = true;
        }
        q.intersectSize = S_cap;
        if (S_cap < constraint) {
            return (int) (Math.ceil(2 * n_max * Math.log((4 * q.predAttrs.size() * Main.depth *
                    Math.sqrt(Main.maxSize)) / Main.delta)/(Main.maxSize * Main.eps * Main.eps)));
        } else {
            q.thrm33Case2 = true;
            return (int) Math.ceil(S_cap * n_max / Main.maxSize);
        }

    }

    private int getNmax(Kmin[] samples) {
        int n_max = 0;
        for (Kmin sample : samples) {
            if (sample.n > n_max) {
                n_max = sample.n;
            }
        }
        return n_max;
    }

    private int getEstimateKMV(Kmin[] samples) {
        int c=0;
        Iterator<Long> iter = samples[0].sketch.iterator();
        while(iter.hasNext()) {
            boolean found=true;
            Long i = iter.next();
            for (int j=1;j<samples.length;j++)
                if (!samples[j].sketch.contains(i)) {
                    found=false;
                    break;
                    }
            if (found) c++;
            }
        return c;
    }

    private double getAltEstKMV(Kmin[] samples) {
        int numJoins = samples.length;
        int c = 0;
        Iterator<Long> iter = samples[0].sketch.iterator();
        while (iter != null && iter.hasNext()) {
            boolean found = true;
            Long i = iter.next();
            for (int j = 1; j < numJoins; j++) {
                Long otherElement = samples[j].sketch.ceiling(i);
                if (otherElement == null) {
                    found = false;
                    iter = null;
                    break;
                } // not contained
                else if (otherElement.equals(i)) continue; // is contained
                else {
                    iter = samples[0].sketch.tailSet(otherElement).iterator(); // fast forward iter0
                    found = false;
                    break; // but now you need to start from iter.hasNext() again
                }
            }
            if (found) c++;

        }
        return c;
    }

    private double getAltEstKMV(TreeSet<Long>[] samples) {
        int numJoins = samples.length;
        int c = 0;
        Iterator<Long> iter = samples[0].iterator();
        while (iter != null && iter.hasNext()) {
            boolean found = true;
            Long i = iter.next();
            for (int j = 1; j < numJoins; j++) {
                Long otherElement = samples[j].ceiling(i);
                if (otherElement == null) {
                    found = false;
                    iter = null;
                    break;
                } // not contained
                else if (otherElement.equals(i)) continue; // is contained
                else {
                    iter = samples[0].tailSet(otherElement).iterator(); // fast forward iter0
                    found = false;
                    break; // but now you need to start from iter.hasNext() again
                }
            }
            if (found) c++;

        }
        return c;
    }

    private double getRetainAll(Kmin[] samples) {
        double est = 0;
        TreeSet<Long> intersect = new TreeSet<>();

        for (int i = 0; i < samples.length; i++) {
            if (i == 0) {
                intersect.addAll(samples[i].sketch);
            } else {
                intersect.retainAll(samples[i].sketch);
            }
        }
        est = intersect.size();
        return est;
    }

    private double getIntersectedEstimateDS(Query query, Sample[][] samples) {
        DistinctSample intersectSample = null;
        for (int j = 0; j < Main.depth; j++) {
            for (int i = 0; i < query.predAttrs.size(); i++) {
                if (i == 0 && j == 0) {
                    intersectSample = new DistinctSample((DistinctSample) samples[i][j]);
                } else {
                    DistinctSample other = (DistinctSample) samples[i][j];
                    intersectSample = intersectSample.intersect(other);
                }
            }
        }
        assert intersectSample != null;
        DistinctSample estSample = getSizeAndLevel(intersectSample);
        double estimate = Math.pow(2, estSample.sampleLevel + 1) * estSample.intersectSize;
        query.intersectSize = estSample.intersectSize;
        query.sampleLevel = estSample.sampleLevel;
        return estimate;
    }
    public int[][] ns = new int[Main.numAttributes][Main.depth];

    public int[][] Bs = new int[Main.numAttributes][Main.depth];

    public int getNmax() {
        int n_max = 0;
        for (int i = 0; i < Main.numAttributes; i++) {
            for (int j = 0; j < Main.depth; j++) {
                if (ns[i][j] > n_max) {
                    n_max = ns[i][j];
                }
            }
        }
        return n_max;
    }
    public int rangeQuery(Query q) {
        double S_cap = 0;
        int n_max = 0;
        int B_virtual = 0;
        TreeSet<Long>[] samples = new TreeSet[q.predAttrs.size() * Main.depth];
        // Empty ns;
        for (int i = 0; i < Main.numAttributes; i++) {
            for (int j = 0; j < Main.depth; j++) {
                ns[i][j] = 0;
            }
        }
        for (int i = 0; i < q.predAttrs.size(); i++) {
            int attr = q.predAttrs.get(i);
            ArrayList<long[]> rangesList = wrapLogRanges(q.lower[attr], q.upper[attr]);
            TreeSet<Long>[] set = new TreeSet[Main.depth];
            B_virtual = rangesList.size() * Main.maxSize;
            for (int j = 0; j < rangesList.size(); j++) {
                 CountMinDyad cm = CMSketchesRange[q.predAttrs.get(i)][getIndexOfRange(rangesList.get(j))];
                 TreeSet<Long>[] rangeSet = cm.rangeQuery(rangesList.get(j)[0], rangesList.get(j)[1], ns[attr]);
                 for (int d = 0; d < Main.depth; d++) {
                    if (set[d] == null) {
                        set[d] = rangeSet[d];
                    } else {
                        set[d].addAll(rangeSet[d]);
                    }
                 }

            }
            if (Main.depth >= 0) System.arraycopy(set, 0, samples, i * Main.depth, Main.depth);
        }
        n_max = getNmax();
        S_cap = getAltEstKMV(samples); //Instead of getAltEstKMV
        double constraint = 3 * Math.log((4 * q.predAttrs.size() * Main.depth * Math.sqrt(B_virtual))
                / Main.delta)/(Main.eps * Main.eps);
        q.case2Estimate = Math.ceil(S_cap * n_max / B_virtual);
        double BConstraint =  2 * n_max * Math.log((4 * q.predAttrs.size() * Main.depth *
                Math.sqrt(B_virtual)) / Main.delta)/(q.exactAnswer * Main.eps * Main.eps);
        if (B_virtual >= BConstraint) {
            q.ratioCondition = true;
        }
        q.intersectSize = S_cap;
        if (S_cap < constraint) {
            return (int) (Math.ceil(2 * n_max * Math.log((4 * q.predAttrs.size() * Main.depth *
                    Math.sqrt(B_virtual)) / Main.delta)/(B_virtual * Main.eps * Main.eps)));
        } else {
            q.thrm33Case2 = true;
            return (int) Math.ceil(S_cap * n_max / B_virtual);
        }
    }

    private ArrayList<long[]> wrapLogRanges(long low, long up) {
        ArrayList<long[]> temp;
        if (low < 0 || up < 0) {
            System.out.println("Error because low or up < 0: low: " + low + " up: " + up);
            System.exit(1);
        }
        temp = getLogRangesArrList(low + 1, up + 1);
        for (long[] i: temp) {
            i[0] = i[0] - 1L;// - (long) Math.pow(2, Main.dyadicRangeBits - 1);
            i[1] = i[1] - 1L;// - (long) Math.pow(2, Main.dyadicRangeBits - 1);
        }
        return temp;
    }

    private int getIndexOfRange(long[] longs) {
        // get distance between the two elements in longs.
        return (int) (Math.log(longs[1] - longs[0] + 1) / Math.log(2));
    }

    public static ArrayList<long[]> getLogRangesArrList(long startInclusive, long stopInclusive) {
        startInclusive--;stopInclusive--;
        long initDiff=stopInclusive-startInclusive+1;
        ArrayList<long[]> result = new ArrayList<>();
        long totalSum = 0;
        long pow = 1;
        for (int j = 0; j <= Main.dyadicRangeBits - 1; j++) {
            if (startInclusive+pow-1>stopInclusive)
                break;
            else if (startInclusive%(pow*2)==0 && startInclusive+pow-1<=stopInclusive) ;
                // do nothing, increase the power further
            else {
                result.add(new long[]{1+startInclusive, 1+startInclusive + pow - 1});
                totalSum+=pow;
                startInclusive+=pow;
            }
            pow*=2;
        }

        pow= (long) Math.pow(2,Main.dyadicRangeBits);
        for (int j=Main.dyadicRangeBits;j>=0;j--) {
            if (startInclusive%pow==0L && startInclusive+pow-1<=stopInclusive) {
                result.add(new long[]{1+startInclusive, 1+startInclusive + pow - 1});
                totalSum+=pow;
                startInclusive+=pow;
            }
            pow/=2;
        }

        if (totalSum != initDiff) {
            System.err.println("Error - no full coverage");
        }
        return result;
    }




    public DistinctSample getSizeAndLevel(DistinctSample intersectSample) {
        double intersectSize = 0;
        // Search for smallest common sample level (l_m)
        while (true) {
            assert intersectSample != null;
            if (!(intersectSample.sample.containsKey(intersectSample.sampleLevel) && intersectSample.sample.get(intersectSample.sampleLevel).size() == 0))
                break;
            intersectSample.sampleLevel++;
        }
        if (intersectSample.sample.containsKey(intersectSample.sampleLevel)) {
            intersectSample.setIntersectSize(intersectSample.sample.get(intersectSample.sampleLevel).size());
        }
        if (Main.maxLevel < intersectSample.sampleLevel) {
            throw new IllegalStateException("Max level surpassed");
        }
        return intersectSample;
    }

    public double getPlainEstimateDS(Query query, Sample[][] samples) {

        double estimate = Double.POSITIVE_INFINITY;
        for (int j = 0; j < Main.depth; j++) {
            DistinctSample intersectSample = null;
            for (int i = 0; i < query.predAttrs.size(); i++) {
                if (i == 0 ) {
                    intersectSample = new DistinctSample((DistinctSample) samples[i][j]);
                } else {
                    Sample other = samples[i][j];
                    intersectSample = intersectSample.intersect((DistinctSample) other);

                }
            }
            assert intersectSample != null;
            DistinctSample estSample = getSizeAndLevel(intersectSample);
            if (Math.pow(2, estSample.sampleLevel + 1) * estSample.intersectSize < estimate) {
                minCMRow = j;
                estimate = Math.pow(2, estSample.sampleLevel + 1) * estSample.intersectSize;
                query.intersectSize = estSample.intersectSize;
                query.sampleLevel = estSample.sampleLevel;
            }
        }
        return estimate;
    }

    public void reset() {
        for (int i = 0; i < Main.numAttributes; i++) {
            if (hasPredicate[i]) {
                if (Main.rangeQueries) {
                    for (int j = 0; j < Main.dyadicRangeBits + 1; j++) {
                        CMSketchesRange[i][j].reset();
                    }
                } else {
                    CMSketches[i].reset();
                }
            }
        }
    }


}
