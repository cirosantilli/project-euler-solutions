import solvers.«98.py.lean.d».SquareStep
import lang.py.FormalSW.ClassCalls

namespace solvers.«98».squares_with_length
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open PatternBuckets (Bucket View Partition)
open pattern_signature (program number)
open AnagramSquares (firstRoot lastRoot)
open AnagramSearch (squareText)
set_option Elab.async false

def rangeSquares (first count : Nat) : List String := (List.range' first count).map squareText
def squares (width : Nat) : List String := rangeSquares (firstRoot width) (lastRoot width + 1 - firstRoot width)

theorem square_loop (base suffix : Mutable.Heap) (dictionary setAddress : Nat) (buckets : List Bucket)
    (before : List String) (first count : Nat) (env : HeapCalls.Env)
    (view : View (base ++ suffix) dictionary buckets) (partition : Partition signature before buckets)
    (foundSet : (base ++ suffix)[setAddress]? = some (.set (setValues before)))
    (afterDictionary : base.length ≤ dictionary) (afterSet : base.length ≤ setAddress)
    (getDictionary : env "by_pat" = some (.ref dictionary)) (getSet : env "st" = some (.ref setAddress)) :
    ∃ finalEnv extra newBuckets,
      BlockContinues program env [.rangeStepLoop "n" first (first + count) 1 squareBody []] (base ++ suffix) finalEnv (base ++ extra) ∧
      View (base ++ extra) dictionary newBuckets ∧ Partition signature (before ++ rangeSquares first count) newBuckets ∧
      (base ++ extra)[setAddress]? = some (.set (setValues (before ++ rangeSquares first count))) ∧
      (∀ name, name ∉ ["n", "sq", "s", "pat"] → finalEnv name = env name) := by
  induction count generalizing first before suffix buckets env with
  | zero =>
      refine ⟨env, suffix, buckets, ?_, view, ?_, ?_, by intros; rfl⟩
      · exact BlockContinues.range_done (by simp [Calls.rangeContinues]) BlockContinues.nil
      · simpa [rangeSquares] using partition
      · simpa [rangeSquares] using foundSet
  | succ count ih =>
      let ready := setLocal env "n" (number first)
      obtain ⟨stepEnv, stepExtra, stepBuckets, executed, stepView, stepPartition, stepSet, stepKept⟩ :=
        square_step base suffix dictionary setAddress buckets before first ready view partition foundSet afterDictionary afterSet
          (by simpa [ready, setLocal] using getDictionary) (by simpa [ready, setLocal] using getSet) (by simp [ready, setLocal])
      obtain ⟨finalEnv, extra, newBuckets, continued, finalView, finalPartition, finalSet, kept⟩ :=
        ih stepExtra stepBuckets (before ++ [squareText first]) (first + 1) stepEnv stepView stepPartition stepSet
          ((stepKept "by_pat" (by decide)).trans (by simpa [ready, setLocal] using getDictionary))
          ((stepKept "st" (by decide)).trans (by simpa [ready, setLocal] using getSet))
      refine ⟨finalEnv, extra, newBuckets, ?_, finalView, ?_, ?_, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, squareBody,
          Bool.false_or, Bool.false_eq_true, if_false, Int.natCast_add, Int.natCast_one,
          Int.add_assoc, Int.add_comm 1 (count : Int)] using executed.append continued
      · simpa only [rangeSquares, List.range'_succ, List.map_cons, List.append_assoc, List.singleton_append] using finalPartition
      · simpa only [rangeSquares, List.range'_succ, List.map_cons, List.append_assoc, List.singleton_append] using finalSet
      · intro name different
        have notRoot : name ≠ "n" := fun same => different (by simp [same])
        have notStep : name ∉ ["sq", "s", "pat"] := fun member => different (List.mem_cons_of_mem _ member)
        rw [kept name different, stepKept name notStep]
        simp [ready, setLocal, notRoot]

abbrev classInfo := (program.findClass "SquaresByLength").getD ("", [])
abbrev fieldArgs := ((dataclassFields classInfo.2).getD (false, [])).2
def record (dictionary setAddress : Nat) : Mutable.Object :=
  .dataInstance classInfo.1 "SquaresByLength" false ["by_pattern", "as_set"]
    [("by_pattern", .ref dictionary), ("as_set", .ref setAddress)]

theorem construct (heap : Mutable.Heap) (dictionary setAddress : Nat) :
    ReturnsValue program "SquaresByLength" [.ref dictionary, .ref setAddress] heap (.ref heap.length)
      (heap ++ [record dictionary setAddress]) := by
  exact ReturnsValue.dataclass_constructor (classBody := classInfo.2) (fields := fieldArgs) rfl rfl rfl rfl rfl rfl

/-- The complete square-cache helper returns every square of the requested width, partitioned by its canonical pattern. -/
theorem result (width : Nat) (positive : 0 < width) (heap : Mutable.Heap) :
    ∃ extra buckets address,
      ReturnsValue program "squares_with_length" [number width] heap (.ref address) (heap ++ extra) ∧
      View (heap ++ extra) heap.length buckets ∧ Partition signature (squares width) buckets ∧
      (heap ++ extra)[heap.length + 1]? = some (.set (setValues (squares width))) ∧
      (heap ++ extra)[address]? = some (record heap.length (heap.length + 1)) := by
  let source := ((findFunctionIn program.body "squares_with_length").getD ([], [])).2
  let args := ((findFunctionIn program.body "squares_with_length").getD ([], [])).1
  let initial := functionEnv args source [number width] []
  let bounded := ready initial width
  let withDictionary := setLocal bounded "by_pat" (.ref heap.length)
  let entry := setLocal withDictionary "st" (.ref (heap.length + 1))
  have initializedView : View (heap ++ [.dict [], .set []]) heap.length [] := by
    simpa only [List.append_assoc, List.singleton_append] using (PatternBuckets.empty_view heap).append [.set []]
  obtain ⟨finalEnv, extra, buckets, scanned, finalView, partition, finalSet, kept⟩ :=
    square_loop heap [.dict [], .set []] heap.length (heap.length + 1) [] [] (firstRoot width)
      (lastRoot width + 1 - firstRoot width) entry initializedView (Partition.empty signature)
      (by simp [setValues]) (by omega) (by omega) (by simp [entry, withDictionary, setLocal]) (by simp [entry, setLocal])
  have stop : firstRoot width + (lastRoot width + 1 - firstRoot width) = lastRoot width + 1 := by
    have bounds := AnagramSearch.ordered_roots width
    omega
  let finalHeap := heap ++ extra
  have returnedDictionary : finalEnv "by_pat" = some (.ref heap.length) :=
    (kept "by_pat" (by decide)).trans (by simp [entry, withDictionary, setLocal])
  have returnedSet : finalEnv "st" = some (.ref (heap.length + 1)) :=
    (kept "st" (by decide)).trans (by simp [entry, setLocal])
  have returned : BlockReturns program finalEnv [.return (.call (.name "SquaresByLength") [.name "by_pat", .name "st"])]
      finalHeap (.ref finalHeap.length) (finalHeap ++ [record heap.length (heap.length + 1)]) :=
    BlockReturns.return_value (Evaluates.call (.cons (Evaluates.lookup returnedDictionary) (.cons (Evaluates.lookup returnedSet) .nil))
      (construct finalHeap heap.length (heap.length + 1)))
  refine ⟨extra ++ [record heap.length (heap.length + 1)], buckets, finalHeap.length, ?_, ?_, ?_, ?_, ?_⟩
  · have boundedRun := prepare width positive heap initial (by rfl)
    have initialized : BlockContinues program bounded (source.drop 4 |>.take 2) heap entry (heap ++ [.dict [], .set []]) := by
      apply (BlockContinues.annAssign (name := "by_pat") (Evaluates.dictionary .nil)).append
      have created : Evaluates program withDictionary (.builtinCall .set []) (heap ++ [.dict []])
          (.ref (heap.length + 1)) (heap ++ [.dict [], .set []]) := by
        apply Evaluates.builtin .nil
        simp [primitive, Mutable.allocateSet, Except.map]
      exact BlockContinues.annAssign created
    have looped : BlockContinues program entry (source.drop 6 |>.take 1) (heap ++ [.dict [], .set []]) finalEnv finalHeap := by
      apply BlockContinues.for_range (middle := heap ++ [.dict [], .set []])
        (.cons (Evaluates.lookup (show entry "n0" = some (number (firstRoot width)) by rfl))
          (.cons (Evaluates.binary (av := number (lastRoot width)) (bv := number 1)
            (value := number (lastRoot width + 1))
            (Evaluates.lookup (show entry "n1" = some (number (lastRoot width)) by rfl)) Evaluates.constant
            (by simp [Mutable.binary, scalarBinary, number])) .nil)) rfl
      simpa only [← Int.natCast_add, stop] using scanned
    apply BlockReturns.function (args := args) (body := source) rfl rfl
    simpa only [List.append_assoc, finalHeap] using boundedRun.returns (initialized.returns (looped.returns returned))
  · simpa only [List.append_assoc] using finalView.append [record heap.length (heap.length + 1)]
  · simpa only [squares, List.nil_append] using partition
  · simpa only [squares, List.nil_append, finalHeap, List.append_assoc] using
      HeapCollection.prefix_found finalHeap [record heap.length (heap.length + 1)] (heap.length + 1) _ finalSet
  · rw [← List.append_assoc]
    change (finalHeap ++ [record heap.length (heap.length + 1)])[finalHeap.length]? = _
    simp

end solvers.«98».squares_with_length
