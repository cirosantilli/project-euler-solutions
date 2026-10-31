import solvers.«90.py.lean.d».Masks
import ProjectEulerSolutions.Lib.HeapCollection
import lang.py.FormalSW.CombinationCalls

/-! The complete combination collector, including empty tuples and all private allocations. -/
namespace solvers.«90».masks
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions DigitDice
set_option Elab.async false

abbrev working (base : Mutable.Heap) (values : List Nat) (garbage : Mutable.Heap) :=
  HeapCollection.working base (values.map number) garbage

def captured (rows : List (List Nat)) : List (List SequenceSource) :=
  rows.map (fun row => (row.map number).map sequenceSource)

/-- A single body visits every label and appends its mask without modifying caller objects. -/
theorem visit (base : Mutable.Heap) (values : List Nat) (garbage : Mutable.Heap)
    (digits : List Nat) (address : Nat) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : (working base values garbage)[address]? = some (.tuple (digits.map number)))
    (getComb : env "comb" = some (.ref address)) (getMasks : env "masks" = some (.ref base.length)) :
    ∃ finalEnv, BlockContinues program env combinationCode (working base values garbage) finalEnv
      (working base (values ++ [DigitMask.mask digits]) garbage) globals ∧
      (∀ name, name ≠ "m" → name ≠ "d" → finalEnv name = env name) := by
  let reset := setLocal env "m" (number 0)
  have sequence : Mutable.sequence (sequenceValue (.ref address)) (working base values garbage) =
      some (.ok (digits.map number), working base values garbage) := by
    simp [sequenceValue, Mutable.sequence, found]
  obtain ⟨finalEnv, loop, result, preserved⟩ :=
    digit_loop digits digits 0 0 (.ref address) (working base values garbage) globals reset sequence rfl rfl
  have getResult : finalEnv "m" = some (number (DigitMask.mask digits)) := by simpa using result
  have finalMasks : finalEnv "masks" = some (.ref base.length) := by
    rw [preserved "masks" (by decide) (by decide)]
    simpa [reset, setLocal] using getMasks
  have traversed : BlockContinues program reset [combinationCode[1]'(by decide)]
      (working base values garbage) finalEnv (working base values garbage) globals := by
    apply BlockContinues.for_sequence (by intros; simp)
      (Evaluates.lookup (by simpa [reset, setLocal] using getComb)) sequence
    exact loop
  have emitted : BlockContinues program finalEnv [combinationCode[2]'(by decide)]
      (working base values garbage) finalEnv (working base (values ++ [DigitMask.mask digits]) garbage) globals := by
    have appended := Evaluates.append (module := program) (globals := globals)
      (Evaluates.lookup finalMasks) (HeapCollection.found base (values.map number) garbage)
      (Evaluates.lookup getResult) (HeapCollection.found base (values.map number) garbage)
    simpa only [working, List.map_append, List.map_cons, List.map_nil, HeapCollection.store] using
      BlockContinues.discard appended
  refine ⟨finalEnv, ?_, ?_⟩
  · exact (BlockContinues.assign Evaluates.constant).append (traversed.append emitted)
  · intro name notM notD
    rw [preserved name notM notD]
    simp [reset, setLocal, notM]

/-- Every captured combination contributes one mask; equal masks retain their separate positions. -/
theorem collect (rows : List (List Nat)) (base : Mutable.Heap) (values : List Nat)
    (garbage : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getMasks : env "masks" = some (.ref base.length)) :
    ∃ finalEnv finalGarbage, BlockContinues program env
      [.permutationStepLoop "comb" (captured rows) combinationCode []]
      (working base values garbage) finalEnv
      (working base (values ++ rows.map DigitMask.mask) finalGarbage) globals ∧
      (∀ name, name ≠ "comb" → name ≠ "m" → name ≠ "d" → finalEnv name = env name) := by
  induction rows generalizing values garbage env with
  | nil =>
      refine ⟨env, garbage, ?_, by intros; rfl⟩
      simpa [captured] using (BlockContinues.permutation_done (module := program)
        (globals := globals) (env := env) (name := "comb") (body := combinationCode)
        (heap := working base values garbage) BlockContinues.nil)
  | cons digits rest ih =>
      obtain ⟨address, extra, allocated, found⟩ :=
        HeapCollection.allocate_tuple (digits.map number) (working base values garbage)
      let bound := setLocal env "comb" (.ref address)
      have tupleFound : (working base values (garbage ++ extra))[address]? =
          some (.tuple (digits.map number)) := by simpa only [HeapCollection.append] using found
      obtain ⟨visitedEnv, visited, preserved⟩ := visit base values (garbage ++ extra) digits address bound globals
        tupleFound rfl (by simpa [bound, setLocal] using getMasks)
      have masksKept : visitedEnv "masks" = some (.ref base.length) := by
        rw [preserved "masks" (by decide) (by decide)]
        simpa [bound, setLocal] using getMasks
      obtain ⟨finalEnv, finalGarbage, continued, kept⟩ :=
        ih (values ++ [DigitMask.mask digits]) (garbage ++ extra) visitedEnv masksKept
      refine ⟨finalEnv, finalGarbage, ?_, ?_⟩
      · apply BlockContinues.permutation_step (tuple := .ref address)
          (middle := working base values (garbage ++ extra))
        · simpa [captured, List.map_map, Function.comp_def, sequenceValue, sequenceSource,
            HeapCollection.append] using allocated
        · simpa only [List.map_cons, List.append_assoc, List.singleton_append] using visited.append continued
      · intro name notComb notM notD
        rw [kept name notComb notM notD, preserved name notM notD]
        simp [bound, setLocal, notComb]

def poolExpression : Expr := .listComp (.name "$formalsw.range.item")
  [("$formalsw.range.item", .call (.name "range") [.constant (.int 10)], [])]

/-- The private range snapshot has all ten decimal labels in increasing order. -/
theorem pool_result (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals) :
    ∃ extra, Evaluates program env poolExpression heap (.ref heap.length)
      (heap ++ (.list ((List.range 10).map number) :: extra)) globals := by
  obtain ⟨extra, collected⟩ := HeapCollection.range_map (globals := globals)
    program "$formalsw.range.item" (.name "$formalsw.range.item") number
    (fun heap env value found => ⟨[], by simpa using
      (Evaluates.lookup (module := program) (globals := globals) (heap := heap) found)⟩)
    heap env [.constant (.int 10)] [number 10] 0 10
    (.cons Evaluates.constant .nil) rfl
  exact ⟨extra, collected⟩

/-- The complete source `for comb in combinations(range(10), face_count)` collector. -/
theorem enumeration (faceCount : Nat) (base : Mutable.Heap) (values : List Nat)
    (garbage : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getCount : env "face_count" = some (number faceCount))
    (getMasks : env "masks" = some (.ref base.length)) :
    ∃ finalEnv finalGarbage, BlockContinues program env [code[2]'(by decide)]
      (working base values garbage) finalEnv
      (working base (values ++ (choices faceCount (List.range 10)).map DigitMask.mask) finalGarbage) globals ∧
      (∀ name, name ≠ "comb" → name ≠ "m" → name ≠ "d" → finalEnv name = env name) := by
  let heap := working base values garbage
  obtain ⟨extra, pool⟩ := pool_result heap env globals
  let added := .list ((List.range 10).map number) :: extra
  let afterPool := working base values (garbage ++ added)
  have poolFound : afterPool[heap.length]? = some (.list ((List.range 10).map number)) := by
    change (HeapCollection.working base (values.map number) (garbage ++ added))[heap.length]? = _
    rw [← HeapCollection.append]
    simp [heap, added]
  have acquired : combinationRows [.ref heap.length, number faceCount] afterPool =
      some (.ok (captured (choices faceCount (List.range 10))), afterPool) := by
    simpa only [captured, choices_rows, Combinations.rows_map, List.map_map, Function.comp_def] using
      combinationRows_list afterPool heap.length faceCount ((List.range 10).map number) poolFound
  obtain ⟨finalEnv, finalGarbage, executed, preserved⟩ := collect
    (choices faceCount (List.range 10)) base values (garbage ++ added) env globals getMasks
  refine ⟨finalEnv, finalGarbage, ?_, preserved⟩
  apply BlockContinues.for_combinations (values := [.ref heap.length, number faceCount])
    (middle := afterPool)
  · refine .cons ?_ (.cons (Evaluates.lookup getCount) .nil)
    simpa only [afterPool, added, heap, working, HeapCollection.append] using pool
  · exact acquired
  · exact executed

end solvers.«90».masks
