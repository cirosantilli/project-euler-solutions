import ProjectEulerSolutions.Lib.HeapCollection

/-! Source writes and bounded range fills for private list tables. -/
namespace ProjectEulerSolutions.HeapTables
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
variable {globals : Mutable.Globals}

/-- The list analogue of the existing buffer write rule. -/
theorem setItem_list (heap : Mutable.Heap) (address position : Nat) (items : List Mutable.Value)
    (replacement : Mutable.Value) (found : heap[address]? = some (.list items)) (inside : position < items.length) :
    Mutable.setItem (.ref address) (.scalar (.int position)) replacement heap =
      some (.ok (.scalar .none), heap.set address (.list (items.set position replacement))) := by
  have nonnegative : ¬ (position : Int) < 0 := by omega
  have inRange : ¬ items.length ≤ position := by omega
  simp [Mutable.setItem, found, Mutable.index, nonnegative, inRange, Except.map]

theorem store (base : Mutable.Heap) (items : List Mutable.Value) (garbage : Mutable.Heap)
    (position : Nat) (replacement : Mutable.Value) (inside : position < items.length) :
    Mutable.setItem (.ref base.length) (.scalar (.int position)) replacement
      (HeapCollection.working base items garbage) =
      some (.ok (.scalar .none), HeapCollection.working base (items.set position replacement) garbage) := by
  simpa only [HeapCollection.store] using
    setItem_list (HeapCollection.working base items garbage) base.length position items replacement
      (HeapCollection.found base items garbage) inside

/-- Mathematical contents after the exact sequence of source writes. -/
def fillRange : Nat → Nat → List Mutable.Value → Mutable.Value → List Mutable.Value
  | 0, _, items, _ => items
  | count + 1, first, items, replacement =>
      fillRange count (first + 1) (items.set first replacement) replacement

@[simp] theorem fillRange_length (count first : Nat) (items : List Mutable.Value) (replacement : Mutable.Value) :
    (fillRange count first items replacement).length = items.length := by
  induction count generalizing first items with
  | zero => rfl
  | succ count ih => simp only [fillRange, ih, List.length_set]

/-- A range fill changes precisely the requested indices; no 256-element source execution is normalized. -/
theorem fillRange_getElem? (count first index : Nat) (items : List Mutable.Value) (replacement : Mutable.Value)
    (inside : first + count ≤ items.length) :
    (fillRange count first items replacement)[index]? =
      if first ≤ index ∧ index < first + count then some replacement else items[index]? := by
  induction count generalizing first items with
  | zero =>
      have outside : ¬(first ≤ index ∧ index < first + 0) := by omega
      simp only [fillRange, if_neg outside]
  | succ count ih =>
      rw [fillRange, ih (first + 1) (items.set first replacement) (by simpa only [List.length_set] using (show first + 1 + count ≤ items.length by omega))]
      by_cases same : index = first
      · subst index
        have valid : first < items.length := by omega
        rw [if_neg (show ¬(first + 1 ≤ first ∧ first < first + 1 + count) by omega),
          List.getElem?_set_self valid,
          if_pos (show first ≤ first ∧ first < first + (count + 1) by omega)]
      · rw [List.getElem?_set_ne (Ne.symm same)]
        by_cases later : first + 1 ≤ index ∧ index < first + 1 + count
        · have current : first ≤ index ∧ index < first + (count + 1) := by omega
          simp only [if_pos later, if_pos current]
        · have current : ¬(first ≤ index ∧ index < first + (count + 1)) := by omega
          simp only [if_neg later, if_neg current]

def fillBody (tableName indexName : String) (replacement : Constant) : List Stmt :=
  [.assign [.subscript (.name tableName) (.name indexName)] (.constant replacement)]

/-- One actual subscription assignment leaves the private table address and caller prefix unchanged. -/
theorem write_index (program : Module) (tableName indexName : String) (replacement : Constant)
    (base : Mutable.Heap) (items : List Mutable.Value) (garbage : Mutable.Heap) (position : Nat)
    (inside : position < items.length) (env : HeapCalls.Env)
    (getTable : env tableName = some (.ref base.length))
    (getIndex : env indexName = some (.scalar (.int position))) :
    BlockContinues program env (fillBody tableName indexName replacement)
      (HeapCollection.working base items garbage) env
      (HeapCollection.working base (items.set position (.scalar replacement)) garbage) globals := by
  apply BlockContinues.setItem (value := .scalar replacement) (receiver := .ref base.length)
    (subscript := .scalar (.int position)) (by intros; simp) Evaluates.constant
    (.cons (Evaluates.lookup getTable) (.cons (Evaluates.lookup getIndex) .nil))
  exact store base items garbage position (.scalar replacement) inside

/-- The unchanged source range body writes each live index once and preserves every other local. -/
theorem range_loop (program : Module) (tableName indexName : String) (different : tableName ≠ indexName)
    (replacement : Constant) (count first : Nat) (stop : Int) (equation : stop = ((first + count : Nat) : Int))
    (base : Mutable.Heap) (items : List Mutable.Value) (garbage : Mutable.Heap)
    (inside : first + count ≤ items.length) (env : HeapCalls.Env)
    (getTable : env tableName = some (.ref base.length)) :
    ∃ afterEnv,
      BlockContinues program env
        [.rangeStepLoop indexName first stop 1 (fillBody tableName indexName replacement) []]
        (HeapCollection.working base items garbage) afterEnv
        (HeapCollection.working base (fillRange count first items (.scalar replacement)) garbage) globals ∧
      ∀ name, name ≠ indexName → afterEnv name = env name := by
  induction count generalizing first items env with
  | zero =>
      refine ⟨env, ?_, fun _ _ => rfl⟩
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) BlockContinues.nil
  | succ count ih =>
      let bound := HeapCalls.setLocal env indexName (.scalar (.int first))
      have boundTable : bound tableName = some (.ref base.length) := by
        simpa [bound, HeapCalls.setLocal, different] using getTable
      have boundIndex : bound indexName = some (.scalar (.int first)) := by simp [bound, HeapCalls.setLocal]
      have written := write_index (globals := globals) program tableName indexName replacement base items garbage first (by omega)
        bound boundTable boundIndex
      obtain ⟨afterEnv, rest, stable⟩ := ih (first + 1) (by omega) (items.set first (.scalar replacement))
        (by simpa only [List.length_set] using (show first + 1 + count ≤ items.length by omega)) bound boundTable
      refine ⟨afterEnv, ?_, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa only [fillRange, fillBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          Bool.false_or, Bool.false_eq_true, if_false, Int.natCast_add, Int.natCast_one] using written.append rest
      · intro name notIndex
        exact (stable name notIndex).trans (by simp [bound, HeapCalls.setLocal, notIndex])

/-- Source range bounds may themselves be real builtin calls, such as ord("0") and ord("9"). -/
theorem range_fill (program : Module) (tableName indexName : String) (different : tableName ≠ indexName)
    (replacement : Constant) (count first : Nat) (stop : Int) (equation : stop = ((first + count : Nat) : Int))
    (base : Mutable.Heap) (items : List Mutable.Value) (garbage : Mutable.Heap)
    (inside : first + count ≤ items.length) (env : HeapCalls.Env)
    (getTable : env tableName = some (.ref base.length)) (expressions : List Expr)
    (arguments : Evaluates.Arguments program env expressions (HeapCollection.working base items garbage)
      [.scalar (.int first), .scalar (.int stop)] (HeapCollection.working base items garbage) globals) :
    ∃ afterEnv,
      BlockContinues program env [.forIn (.name indexName) (.call (.name "range") expressions)
        (fillBody tableName indexName replacement) []] (HeapCollection.working base items garbage) afterEnv
        (HeapCollection.working base (fillRange count first items (.scalar replacement)) garbage) globals ∧
      ∀ name, name ≠ indexName → afterEnv name = env name := by
  obtain ⟨afterEnv, ran, stable⟩ := range_loop program tableName indexName different replacement count first stop
    equation base items garbage inside env getTable
  exact ⟨afterEnv, BlockContinues.for_range arguments rfl ran, stable⟩

end ProjectEulerSolutions.HeapTables

namespace ProjectEulerSolutions.HeapTables
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
variable {globals : Mutable.Globals}

def byteFold (bytes : List UInt8) (update : UInt8 → List Mutable.Value → List Mutable.Value)
    (items : List Mutable.Value) : List Mutable.Value :=
  bytes.foldl (fun table byte => update byte table) items

/-- Immutable-byte iteration retains the captured source and allocates no iterator objects.
The source cell is an actual block proof and must preserve all locals. -/
theorem bytes_loop {base garbage : Mutable.Heap} (program : Module) (tableName indexName : String) (different : tableName ≠ indexName)
    (body : List Stmt) (noControl : Calls.loopControl body = false) (update : UInt8 → List Mutable.Value → List Mutable.Value)
    (invariant : List Mutable.Value → Prop)
    (preserved : ∀ byte items, invariant items → invariant (update byte items))
    (cell : ∀ byte items (env : HeapCalls.Env), invariant items →
      env tableName = some (.ref base.length) →
      env indexName = some (.scalar (.int byte.toNat)) →
      BlockContinues program env body (HeapCollection.working base items garbage) env
        (HeapCollection.working base (update byte items) garbage) globals)
    (before remaining : List UInt8) (items : List Mutable.Value) (safe : invariant items)
    (env : HeapCalls.Env) (getTable : env tableName = some (.ref base.length)) :
    ∃ afterEnv,
      BlockContinues program env
        [.sequenceStepLoop indexName (.scalar (.bytes (before ++ remaining))) before.length body []]
        (HeapCollection.working base items garbage) afterEnv
        (HeapCollection.working base (byteFold remaining update items) garbage) globals ∧
      ∀ name, name ≠ indexName → afterEnv name = env name := by
  induction remaining generalizing before items env with
  | nil =>
      refine ⟨env, ?_, fun _ _ => rfl⟩
      apply BlockContinues.sequence_done (items := before.map (fun byte => .scalar (.int byte.toNat)))
      · simp [sequenceValue, Mutable.sequence]
      · simp
      · exact BlockContinues.nil
  | cons byte remaining ih =>
      let bound := HeapCalls.setLocal env indexName (.scalar (.int byte.toNat))
      have boundTable : bound tableName = some (.ref base.length) := by
        simpa [bound, HeapCalls.setLocal, different] using getTable
      have boundIndex : bound indexName = some (.scalar (.int byte.toNat)) := by simp [bound, HeapCalls.setLocal]
      have written := cell byte items bound safe boundTable boundIndex
      obtain ⟨afterEnv, rest, stable⟩ := ih (before ++ [byte]) (update byte items)
        (preserved byte items safe) bound boundTable
      refine ⟨afterEnv, ?_, ?_⟩
      · apply BlockContinues.sequence_step
          (items := (before ++ byte :: remaining).map (fun item => .scalar (.int item.toNat)))
          (value := .scalar (.int byte.toNat))
        · rfl
        · rw [List.getElem?_map, List.getElem?_append_right (Nat.le_refl _)]
          simp
        · simpa only [byteFold, List.foldl_cons, List.length_append, List.length_singleton,
            List.append_assoc, List.singleton_append, Calls.loopContinuation, noControl, Bool.false_eq_true, if_false] using written.append rest
      · intro name notIndex
        exact (stable name notIndex).trans (by simp [bound, HeapCalls.setLocal, notIndex])

/-- Acquire the immutable bytes once, then execute the real per-byte source body. -/
theorem bytes_result {base garbage : Mutable.Heap} (program : Module) (tableName indexName : String) (different : tableName ≠ indexName)
    (body : List Stmt) (noControl : Calls.loopControl body = false) (update : UInt8 → List Mutable.Value → List Mutable.Value)
    (invariant : List Mutable.Value → Prop)
    (preserved : ∀ byte items, invariant items → invariant (update byte items))
    (cell : ∀ byte items (env : HeapCalls.Env), invariant items →
      env tableName = some (.ref base.length) →
      env indexName = some (.scalar (.int byte.toNat)) →
      BlockContinues program env body (HeapCollection.working base items garbage) env
        (HeapCollection.working base (update byte items) garbage) globals)
    (bytes : List UInt8) (items : List Mutable.Value) (safe : invariant items)
    (env : HeapCalls.Env) (getTable : env tableName = some (.ref base.length)) :
    ∃ afterEnv,
      BlockContinues program env [.forIn (.name indexName) (.constant (.bytes bytes)) body []]
        (HeapCollection.working base items garbage) afterEnv
        (HeapCollection.working base (byteFold bytes update items) garbage) globals ∧
      ∀ name, name ≠ indexName → afterEnv name = env name := by
  obtain ⟨afterEnv, executed, stable⟩ := bytes_loop program tableName indexName different body noControl update invariant
    preserved cell [] bytes items safe env getTable
  refine ⟨afterEnv, ?_, stable⟩
  apply BlockContinues.for_sequence (by intro expressions; simp) Evaluates.constant (by rfl)
  exact executed

end ProjectEulerSolutions.HeapTables
