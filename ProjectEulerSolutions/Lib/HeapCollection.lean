import lang.py.FormalSW.PermutationCalls

/-! Heap layouts for fresh comprehension collectors and temporary permutation tuples. -/
namespace ProjectEulerSolutions.HeapCollection
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

set_option Elab.async false

variable {globals : Mutable.Globals}

def working (base : Mutable.Heap) (values : List Mutable.Value) (garbage : Mutable.Heap) :
    Mutable.Heap := base ++ (.list values :: garbage)

@[simp] theorem found (base : Mutable.Heap) (values : List Mutable.Value) (garbage : Mutable.Heap) :
    (working base values garbage)[base.length]? = some (.list values) := by simp [working]

theorem append (base : Mutable.Heap) (values : List Mutable.Value) (garbage extra : Mutable.Heap) :
    working base values garbage ++ extra = working base values (garbage ++ extra) := by
  simp [working, List.append_assoc]

theorem store (base : Mutable.Heap) (values replacement : List Mutable.Value) (garbage : Mutable.Heap) :
    (working base values garbage).set base.length (.list replacement) = working base replacement garbage := by
  simp [working, List.set_append_right]

theorem prefix_found (base extra : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (found : base[address]? = some object) : (base ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

/-- Changing a collector and appending temporaries preserves every other existing object. -/
theorem other_found (base : Mutable.Heap) (old replacement : List Mutable.Value) (garbage extra : Mutable.Heap)
    (address : Nat) (object : Mutable.Object) (different : base.length ≠ address)
    (original : (working base old garbage)[address]? = some object) :
    (working base replacement (garbage ++ extra))[address]? = some object := by
  have kept := prefix_found (working base old garbage) extra address object original
  rw [append] at kept
  have written := List.getElem?_set_ne (l := working base old (garbage ++ extra))
    (a := .list replacement) different
  rw [store] at written
  exact written.trans kept

/-- Empty tuples can be interned; every tuple allocation preserves the original heap prefix. -/
theorem allocate_tuple (items : List Mutable.Value) (heap : Mutable.Heap) :
    ∃ address extra, Mutable.allocateTuple items heap = some (.ok (.ref address), heap ++ extra) ∧
      (heap ++ extra)[address]? = some (.tuple items) := by
  cases items with
  | nil =>
      cases selected : heap.findIdx? (fun object => decide (object = .tuple [])) with
      | none =>
          exact ⟨heap.length, [.tuple []], by simp [Mutable.allocateTuple, selected, Except.map], by simp⟩
      | some address =>
          have found := List.of_findIdx?_eq_some selected
          cases entry : heap[address]? with
          | none => simp [entry] at found
          | some object =>
              have same : object = .tuple [] := of_decide_eq_true (by simpa only [entry] using found)
              subst object
              exact ⟨address, [], by simp [Mutable.allocateTuple, selected], by simpa using entry⟩
  | cons head tail =>
      exact ⟨heap.length, [.tuple (head::tail)], by simp [Mutable.allocateTuple, Except.map], by simp⟩

theorem emit {program : Module} {env : HeapCalls.Env} {expression : Expr}
    (base : Mutable.Heap) (values : List Mutable.Value) (garbage extra : Mutable.Heap)
    (value : Mutable.Value)
    (evaluated : Evaluates (globals := globals) program env expression (working base values garbage) value
      (working base values garbage ++ extra)) :
    BlockContinues (globals := globals) program env [.comprehensionAppend base.length expression]
      (working base values garbage) env (working base (values ++ [value]) (garbage ++ extra)) := by
  have executed := BlockContinues.comprehension_append evaluated
    (by rw [append]; exact found base values (garbage ++ extra))
  simpa only [append, store] using executed

private theorem range_loop (program : Module) (name : String) (element : Expr) (encode : Nat → Mutable.Value)
    (callback : ∀ (heap : Mutable.Heap) (env : HeapCalls.Env) (n : Nat),
      env name = some (.scalar (.int n)) →
      ∃ extra, Evaluates (globals := globals) program env element heap (encode n) (heap ++ extra))
    (first count stop : Nat) (length : first + count = stop)
    (base : Mutable.Heap) (values : List Mutable.Value) (garbage : Mutable.Heap) (env : HeapCalls.Env) :
    ∃ finalEnv finalGarbage, BlockContinues (globals := globals) program env
      [.rangeStepLoop name first stop 1 [.comprehensionAppend base.length element] []]
      (working base values garbage) finalEnv
      (working base (values ++ (List.range' first count).map encode) finalGarbage) := by
  induction count generalizing first values garbage env with
  | zero =>
      refine ⟨env, garbage, ?_⟩
      simp only [List.range'_zero, List.map_nil, List.append_nil]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      let bound := setLocal env name (.scalar (.int first))
      obtain ⟨extra, evaluated⟩ := callback (working base values garbage) bound first (by simp [bound, setLocal])
      have emitted := emit base values garbage extra (encode first) evaluated
      obtain ⟨finalEnv, finalGarbage, rest⟩ := ih (first+1) (by omega)
        (values ++ [encode first]) (garbage ++ extra) bound
      refine ⟨finalEnv, finalGarbage, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
        Bool.false_or, Bool.false_eq_true, if_false, List.range'_succ, List.map_cons,
        List.append_assoc, List.singleton_append, Int.natCast_add, Int.natCast_one] using emitted.append rest

/-- Source callbacks may allocate temporary lists; they must preserve the input heap prefix. -/
theorem range_map_effectful (program : Module) (name : String) (element : Expr) (encode : Nat → Mutable.Value)
    (callback : ∀ (heap : Mutable.Heap) (env : HeapCalls.Env) (n : Nat),
      env name = some (.scalar (.int n)) →
      ∃ extra, Evaluates (globals := globals) program env element heap (encode n) (heap ++ extra))
    (heap middle : Mutable.Heap) (env : HeapCalls.Env) (expressions : List Expr) (values : List Mutable.Value)
    (first count : Nat)
    (arguments : Evaluates.Arguments (globals := globals) program env expressions heap values middle)
    (bounds : rangeBounds values = .ok (first, first+count, 1)) :
    ∃ extra, Evaluates (globals := globals) program env
      (.listComp element [(name, .call (.name "range") expressions, [])]) heap (.ref middle.length)
      (working middle ((List.range' first count).map encode) extra) := by
  obtain ⟨finalEnv, extra, executed⟩ := range_loop program name element encode callback
    first count (first+count) rfl middle [] []
      (comprehensionEnv env [(name, .call (.name "range") expressions, [])])
  refine ⟨extra, ?_⟩
  apply Evaluates.list_comp_range arguments bounds
  simpa only [working, List.nil_append] using executed

/-- The unchanged-heap specialization for pure range bounds. -/
theorem range_map (program : Module) (name : String) (element : Expr) (encode : Nat → Mutable.Value)
    (callback : ∀ (heap : Mutable.Heap) (env : HeapCalls.Env) (n : Nat),
      env name = some (.scalar (.int n)) →
      ∃ extra, Evaluates (globals := globals) program env element heap (encode n) (heap ++ extra))
    (heap : Mutable.Heap) (env : HeapCalls.Env) (expressions : List Expr) (values : List Mutable.Value)
    (first count : Nat)
    (arguments : Evaluates.Arguments (globals := globals) program env expressions heap values heap)
    (bounds : rangeBounds values = .ok (first, first+count, 1)) :
    ∃ extra, Evaluates (globals := globals) program env
      (.listComp element [(name, .call (.name "range") expressions, [])]) heap (.ref heap.length)
      (working heap ((List.range' first count).map encode) extra) :=
  range_map_effectful program name element encode callback heap heap env expressions values first count arguments bounds

end ProjectEulerSolutions.HeapCollection
