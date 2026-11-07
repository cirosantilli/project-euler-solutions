import solvers.«98.py.lean.d».CacheCollect
import solvers.«98.py.lean.d».WordStorage

namespace solvers.«98».square_cache
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open pattern_signature (program)
open squares_with_length (signature squares setValues record classInfo)
set_option Elab.async false

theorem read (heap : Mutable.Heap) (outer : Nat) (rows : Rows) (width : Int) (env : HeapCalls.Env)
    (found : heap[outer]? = some (.dict (KeyedReferences.entries KeyedReferences.integers rows)))
    (valid : Valid outer heap rows) (present : (KeyedReferences.lookup rows width).isSome)
    (getCache : env "squares_cache" = some (.ref outer)) (getWidth : env "L" = some (number width)) :
    ∃ address, Evaluates program env (.subscript (.name "squares_cache") (.name "L")) heap (.ref address) heap ∧
      Stored outer heap width address := by
  cases selected : KeyedReferences.lookup rows width with
  | none => simp [selected] at present
  | some address =>
      refine ⟨address, ?_, valid (width, address) (KeyedReferences.lookup_mem rows width address selected)⟩
      apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getCache) (Evaluates.lookup getWidth)
      apply Mutable.getItem_dictionary heap outer (KeyedReferences.entries KeyedReferences.integers rows) found (.int width)
      simpa only [selected, Option.map_some] using KeyedReferences.lookup_entries KeyedReferences.integers rows width

/-- Reading the actual dataclass fields yields the shared pattern dictionary and square set. -/
theorem fields (heap : Mutable.Heap) (outer : Nat) (width : Int) (address : Nat) (env : HeapCalls.Env)
    (stored : Stored outer heap width address) (getRecord : env "sqinfo" = some (.ref address)) :
    ∃ dictionary buckets,
      Evaluates program env (.attribute (.name "sqinfo") "by_pattern") heap (.ref dictionary) heap ∧
      Evaluates program env (.attribute (.name "sqinfo") "as_set") heap (.ref (dictionary + 1)) heap ∧
      PatternBuckets.View heap dictionary buckets ∧ PatternBuckets.Partition signature (squares width.toNat) buckets ∧
      heap[dictionary + 1]? = some (.set (setValues (squares width.toNat))) := by
  obtain ⟨dictionary, buckets, _, view, partition, foundSet, foundRecord⟩ := stored
  have defined : program.findClass "SquaresByLength" = some classInfo := rfl
  refine ⟨dictionary, buckets, ?_, ?_, view, partition, foundSet⟩
  · apply Evaluates.field (Evaluates.lookup getRecord)
    simp [getAttribute, foundRecord, record, instanceClass, defined, List.lookup]
  · apply Evaluates.field (Evaluates.lookup getRecord)
    simp [getAttribute, foundRecord, record, instanceClass, defined, List.lookup]

end solvers.«98».square_cache

namespace solvers.«98».word_patterns
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open pattern_signature (program)
set_option Elab.async false

theorem read (heap : Mutable.Heap) (outer : Nat) (rows : Rows) (word : String) (env : HeapCalls.Env)
    (found : heap[outer]? = some (.dict (KeyedReferences.entries KeyedReferences.strings rows)))
    (valid : Valid heap rows) (present : (KeyedReferences.lookup rows word).isSome)
    (getCache : env "word_pat" = some (.ref outer)) (getWord : env "w1" = some (text word)) :
    ∃ address, Evaluates program env (.subscript (.name "word_pat") (.name "w1")) heap (.ref address) heap ∧
      heap[address]? = some (tuple word) := by
  cases selected : KeyedReferences.lookup rows word with
  | none => simp [selected] at present
  | some address =>
      refine ⟨address, ?_, valid (word, address) (KeyedReferences.lookup_mem rows word address selected)⟩
      apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getCache) (Evaluates.lookup getWord)
      apply Mutable.getItem_dictionary heap outer (KeyedReferences.entries KeyedReferences.strings rows) found (.string word)
      simpa only [selected, Option.map_some] using KeyedReferences.lookup_entries KeyedReferences.strings rows word

end solvers.«98».word_patterns
