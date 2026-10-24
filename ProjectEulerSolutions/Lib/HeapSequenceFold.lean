import ProjectEulerSolutions.Lib.HeapRangeFold
import ProjectEulerSolutions.Lib.HeapCollection

namespace ProjectEulerSolutions.HeapSequenceFold
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
open HeapRangeFold
set_option Elab.async false
universe u
variable {globals : Mutable.Globals}

/-- Fold an actual source sequence loop whose body can allocate and update
locals while preserving the existing heap. The iterable is read by index on
each iteration, and its original object remains live throughout nested calls. -/
theorem loop {State : Type u} (program : Module) (name : String) (body : List Stmt)
    (noControl : Calls.loopControl body = false) (changed : List String) (written : name ∈ changed)
    (encode : Nat → Mutable.Value) (update : State → Nat → State)
    (Invariant : Mutable.Heap → HeapCalls.Env → State → Prop)
    (address : Nat) (values : List Nat)
    (bindInvariant : ∀ heap env state value, Invariant heap env state → value ∈ values →
      Invariant heap (setLocal env name (encode value)) state)
    (cell : ∀ heap env state value, Invariant heap env state → value ∈ values → env name = some (encode value) →
      ∃ afterEnv finalHeap, BlockContinues program env body heap afterEnv finalHeap globals ∧
        Invariant finalHeap afterEnv (update state value) ∧
        (∃ extra, finalHeap = heap ++ extra) ∧ Frame changed env afterEnv)
    (remaining : List Nat) (position : Nat) (suffix : values.drop position = remaining) (within : position ≤ values.length)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (state : State) (safe : Invariant heap env state)
    (found : heap[address]? = some (.list (values.map encode))) :
    ∃ afterEnv finalHeap,
      BlockContinues program env [.sequenceStepLoop name (.ref address) position body []] heap afterEnv finalHeap globals ∧
      Invariant finalHeap afterEnv (remaining.foldl update state) ∧
      (∃ extra, finalHeap = heap ++ extra) ∧ Frame changed env afterEnv := by
  induction remaining generalizing position heap env state with
  | nil =>
    have length := congrArg List.length suffix
    simp only [List.length_drop, List.length_nil] at length
    have positionEq : position = values.length := by omega
    refine ⟨env, heap, ?_, safe, ⟨[], by simp⟩, Frame.refl changed env⟩
    exact BlockContinues.sequence_done (items := values.map encode) (by simp [sequenceValue, Mutable.sequence, found])
      (by simp [positionEq]) .nil
  | cons value rest ih =>
    have item : values[position]? = some value := by
      have head : (values.drop position)[0]? = some value := by rw [suffix]; rfl
      simpa using head
    have member : value ∈ values := List.mem_of_getElem? item
    let bound := setLocal env name (encode value)
    have boundSafe := bindInvariant heap env state value safe member
    obtain ⟨middleEnv, middleHeap, executed, middleSafe, ⟨firstExtra, firstHeap⟩, firstFrame⟩ :=
      cell heap bound state value boundSafe member (by simp [bound, setLocal])
    have middleFound : middleHeap[address]? = some (.list (values.map encode)) := by
      rw [firstHeap]
      exact HeapCollection.prefix_found heap firstExtra address _ found
    have tailSuffix : values.drop (position + 1) = rest := by
      simpa [List.drop_drop, Nat.add_comm] using congrArg (List.drop 1) suffix
    have nextWithin : position + 1 ≤ values.length := by
      have valid := (List.getElem?_eq_some_iff.mp item).1
      omega
    obtain ⟨afterEnv, finalHeap, continued, finalSafe, ⟨restExtra, restHeap⟩, restFrame⟩ :=
      ih (position + 1) tailSuffix nextWithin middleHeap middleEnv (update state value) middleSafe middleFound
    refine ⟨afterEnv, finalHeap, ?_, finalSafe, ?_, ?_⟩
    · apply BlockContinues.sequence_step (items := values.map encode) (value := encode value) (by simp [sequenceValue, Mutable.sequence, found])
        (by simpa only [List.getElem?_map] using congrArg (Option.map encode) item)
      simpa only [Calls.loopContinuation, noControl, Bool.false_eq_true, if_false] using executed.append continued
    · exact ⟨firstExtra ++ restExtra, by rw [restHeap, firstHeap, List.append_assoc]⟩
    · exact (Frame.updated changed env name _ written).trans (firstFrame.trans restFrame)

/-- The source evaluates the iterable once, then runs the sequence fold above. -/
theorem for_list {State : Type u} (program : Module) (name : String) (body : List Stmt)
    (noControl : Calls.loopControl body = false) (changed : List String) (written : name ∈ changed)
    (encode : Nat → Mutable.Value) (update : State → Nat → State)
    (Invariant : Mutable.Heap → HeapCalls.Env → State → Prop) (address : Nat) (values : List Nat)
    (bindInvariant : ∀ heap env state value, Invariant heap env state → value ∈ values →
      Invariant heap (setLocal env name (encode value)) state)
    (cell : ∀ heap env state value, Invariant heap env state → value ∈ values → env name = some (encode value) →
      ∃ afterEnv finalHeap, BlockContinues program env body heap afterEnv finalHeap globals ∧
        Invariant finalHeap afterEnv (update state value) ∧
        (∃ extra, finalHeap = heap ++ extra) ∧ Frame changed env afterEnv)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (state : State) (safe : Invariant heap env state)
    (iterable : Expr) (ordinary : ∀ expressions, iterable ≠ .call (.name "range") expressions)
    (evaluated : Evaluates program env iterable heap (.ref address) heap globals)
    (found : heap[address]? = some (.list (values.map encode))) :
    ∃ afterEnv finalHeap,
      BlockContinues program env [.forIn (.name name) iterable body []] heap afterEnv finalHeap globals ∧
      Invariant finalHeap afterEnv (values.foldl update state) ∧
      (∃ extra, finalHeap = heap ++ extra) ∧ Frame changed env afterEnv := by
  obtain ⟨afterEnv, finalHeap, executed, safe, extra, frame⟩ := loop program name body noControl changed written encode update
    Invariant address values bindInvariant cell values 0 (by simp) (by omega) heap env state safe found
  refine ⟨afterEnv, finalHeap, ?_, safe, extra, frame⟩
  exact BlockContinues.for_sequence (items := values.map encode) ordinary evaluated (by simp [Mutable.sequence, found]) executed

/-- A nested enumeration loop conditionally runs a proved child block. Guard
evaluation preserves the heap, and skipped items retain the fold state. -/
theorem for_guarded_list {State : Type u} (program : Module) (name : String) (test : Expr) (body : List Stmt)
    (noControl : Calls.loopControl body = false) (changed : List String) (written : name ∈ changed)
    (encode : Nat → Mutable.Value) (accept : Nat → Bool) (update : State → Nat → State)
    (Invariant : Mutable.Heap → HeapCalls.Env → State → Prop) (address : Nat) (values : List Nat)
    (bindInvariant : ∀ heap env state value, Invariant heap env state → value ∈ values →
      Invariant heap (setLocal env name (encode value)) state)
    (guard : ∀ heap env state value, Invariant heap env state → value ∈ values → env name = some (encode value) →
      Evaluates program env test heap (.scalar (.bool (accept value))) heap globals)
    (cell : ∀ heap env state value, Invariant heap env state → value ∈ values → env name = some (encode value) →
      ∃ afterEnv finalHeap, BlockContinues program env body heap afterEnv finalHeap globals ∧
        Invariant finalHeap afterEnv (update state value) ∧
        (∃ extra, finalHeap = heap ++ extra) ∧ Frame changed env afterEnv)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (state : State) (safe : Invariant heap env state)
    (iterable : Expr) (ordinary : ∀ expressions, iterable ≠ .call (.name "range") expressions)
    (evaluated : Evaluates program env iterable heap (.ref address) heap globals)
    (found : heap[address]? = some (.list (values.map encode))) :
    ∃ afterEnv finalHeap,
      BlockContinues program env [.forIn (.name name) iterable [.ifThen test body []] []] heap afterEnv finalHeap globals ∧
      Invariant finalHeap afterEnv (values.foldl (fun state value => if accept value then update state value else state) state) ∧
      (∃ extra, finalHeap = heap ++ extra) ∧ Frame changed env afterEnv := by
  apply for_list program name [.ifThen test body []]
    (by simp [Calls.loopControl, Calls.loopControlStmt, noControl]) changed written encode
    (fun state value => if accept value then update state value else state) Invariant address values bindInvariant
    _ heap env state safe iterable ordinary evaluated found
  intro heap env state value safe member lookup
  have tested := guard heap env state value safe member lookup
  cases selected : accept value with
  | false =>
    refine ⟨env, heap, ?_, ?_, ⟨[], by simp⟩, Frame.refl changed env⟩
    · exact BlockContinues.branch (flag := false) (by simpa only [selected] using tested) rfl .nil
    · simpa only [selected, Bool.false_eq_true, if_false] using safe
  | true =>
    obtain ⟨afterEnv, finalHeap, executed, afterSafe, extra, frame⟩ := cell heap env state value safe member lookup
    refine ⟨afterEnv, finalHeap, ?_, ?_, extra, frame⟩
    · exact BlockContinues.branch (flag := true) (by simpa only [selected] using tested) rfl executed
    · simpa only [selected, if_true] using afterSafe

end ProjectEulerSolutions.HeapSequenceFold
