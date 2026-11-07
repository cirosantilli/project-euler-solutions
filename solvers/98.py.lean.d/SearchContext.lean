import solvers.«98.py.lean.d».GroupProperties

namespace solvers.«98».search_state
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open StringValueBuckets (Bucket)
set_option Elab.async false

/-- The immutable addresses and rows shared by every optimized search iteration. -/
structure Layout where
  buckets : List Bucket
  groupDictionary : Nat
  groupAddress : Nat
  squareAddress : Nat
  patternAddress : Nat
  squareRows : square_cache.Rows
  patternRows : word_patterns.Rows

structure Valid (layout : Layout) (words : List String) (heap : Mutable.Heap) : Prop where
  groupView : StringValueBuckets.View heap layout.groupDictionary layout.buckets
  partition : StringValueBuckets.Partition solve.keyOfValue (solve.groupValues words) layout.buckets
  groupList : heap[layout.groupAddress]? = some (.list ((groups.selected layout.buckets).map groups.reference))
  squareDictionary : heap[layout.squareAddress]? = some (.dict (KeyedReferences.entries KeyedReferences.integers layout.squareRows))
  squareValid : square_cache.Valid layout.squareAddress heap layout.squareRows
  squareCoverage : ∀ width, (KeyedReferences.lookup layout.squareRows width).isSome ↔ width ∈ lengths.needed words layout.buckets
  patternDictionary : heap[layout.patternAddress]? = some (.dict (KeyedReferences.entries KeyedReferences.strings layout.patternRows))
  patternValid : word_patterns.Valid heap layout.patternRows
  patternCoverage : ∀ word, (KeyedReferences.lookup layout.patternRows word).isSome ↔
    word ∈ word_patterns.allWords words (groups.selected layout.buckets)

structure Locals (layout : Layout) (env : HeapCalls.Env) : Prop where
  getGroups : env "anagram_groups" = some (.ref layout.groupAddress)
  getSquares : env "squares_cache" = some (.ref layout.squareAddress)
  getPatterns : env "word_pat" = some (.ref layout.patternAddress)

theorem Valid.append {layout : Layout} {words : List String} {heap : Mutable.Heap}
    (valid : Valid layout words heap) (extra : Mutable.Heap) : Valid layout words (heap ++ extra) :=
  ⟨valid.groupView.append extra, valid.partition,
   HeapCollection.prefix_found heap extra _ _ valid.groupList,
   HeapCollection.prefix_found heap extra _ _ valid.squareDictionary,
   valid.squareValid.append extra, valid.squareCoverage,
   HeapCollection.prefix_found heap extra _ _ valid.patternDictionary,
   valid.patternValid.append extra, valid.patternCoverage⟩

theorem Valid.group_found {layout : Layout} {words : List String} {heap : Mutable.Heap}
    (valid : Valid layout words heap) (bucket : Bucket) (member : bucket ∈ groups.selected layout.buckets) :
    heap[bucket.address]? = some (.list ((lengths.wordsIn words bucket).map word_patterns.text)) := by
  have inside := (List.mem_filter.mp member).1
  rw [← lengths.bucket_values words layout.buckets valid.partition bucket inside]
  exact valid.groupView.lists bucket inside

theorem Valid.pattern_present {layout : Layout} {words : List String} {heap : Mutable.Heap}
    (valid : Valid layout words heap) (bucket : Bucket) (word : String)
    (selected : bucket ∈ groups.selected layout.buckets) (member : word ∈ lengths.wordsIn words bucket) :
    (KeyedReferences.lookup layout.patternRows word).isSome := by
  apply (valid.patternCoverage word).mpr
  exact List.mem_flatMap.mpr ⟨bucket, selected, member⟩

def ofReady {words : List String} {heap : Mutable.Heap} {env : HeapCalls.Env} (ready : solve.Ready words heap env) : Layout :=
  ⟨ready.buckets, ready.groupDictionary, ready.groupAddress, ready.squareAddress, ready.patternAddress, ready.squareRows, ready.patternRows⟩

theorem ready_valid {words : List String} {heap : Mutable.Heap} {env : HeapCalls.Env} (ready : solve.Ready words heap env) :
    Valid (ofReady ready) words heap :=
  ⟨ready.groupView, ready.partition, ready.groupList, ready.squareDictionary, ready.squareValid, ready.squareCoverage,
    ready.patternDictionary, ready.patternValid, ready.patternCoverage⟩

theorem ready_locals {words : List String} {heap : Mutable.Heap} {env : HeapCalls.Env} (ready : solve.Ready words heap env) :
    Locals (ofReady ready) env := ⟨ready.getGroups, ready.getSquares, ready.getPatterns⟩

end solvers.«98».search_state
