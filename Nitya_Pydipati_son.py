
from pyspark import SparkContext
import itertools
import sys
from collections import defaultdict



def candidate_items(frequent_sets,k):
    combine=set()
    if(k>2):
        for sets in frequent_sets:
            for s in sets:
                combine.add(s)
    elif(k==2):
        for sets in frequent_sets:
            combine.add(sets)
    sets=[set(sorted(item)) for item in itertools.chain(*[itertools.combinations(combine, k)])]
    return sets
                
def frequent(frequent_sets,baskets,sup):
    frequent_dict = defaultdict(int)
    for item in frequent_sets:
        for basket in baskets:
            if item.issubset(basket):
                frequent_dict[frozenset(item)] += 1
    items=set()
    for item in frequent_dict:
        if frequent_dict[item] >= sup:
            items.add(item)
    return items
        

def apriori(items,sup):
    freq_one=defaultdict(int)
    baskets=[]
    frequent_sets=set()
    results=dict()
    for item in items:
        baskets.append(item)
        for i in item:
            freq_one[i]+=1
    for freq in freq_one:
        if(freq_one[freq]>=sup):
            frequent_sets.add(freq)
    k=2
    #print frequent_sets
    results[k-1]=[frozenset([item]) for item in frequent_sets]
    while(frequent_sets!=set([])):
        frequent_sets=candidate_items(frequent_sets,k)
        next_freq=frequent(frequent_sets,baskets,sup)
        if (next_freq): 
            results[k]=next_freq
        frequent_sets=next_freq
        k=k+1
    return results

def main():
    sc=SparkContext()
    baskets=sc.textFile(sys.argv[1],2)
    support=float(sys.argv[2])*baskets.count()
    numPartitions=baskets.getNumPartitions()
    outputF=sys.argv[3]
    baskets=baskets.map(lambda x: [int(y) for y in x.split(',')])


    basket=baskets.mapPartitions(lambda line: [y for y in apriori(line,support/numPartitions).values()])

    map_one=basket.flatMap(lambda x: [(y,1) for y in x])

    reduce_one=map_one.reduceByKey(lambda x,y: x)

    item_red=reduce_one.map(lambda (x,y): x).collect()
    broadcasting_global_count=sc.broadcast(item_red)

    map_two=baskets.flatMap(lambda line: [(count,1) for count in broadcasting_global_count.value if set(line).issuperset(set(count))])
    reduce_two=map_two.reduceByKey(lambda x,y: x+y)
#print reduce_two.collect()

    global_count=reduce_two.filter(lambda x: x[1]>=support)
    output=global_count.collect()

    f=open(outputF, 'w')
    for item,count in output:
        item=str(map(int,item)).replace("[","").replace("]","")
    
        f.write("%s\n" % item)

if __name__=="__main__":
    main()    
