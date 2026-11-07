import «.formalsw».solvers.«95.py.ast»
import ProjectEulerSolutions.Lib.HeapTables
import Mathlib.Data.Finset.Lattice.Lemmas

/-! The Boolean processed table and the source loop that marks a completed path. -/
namespace solvers.«95».processed
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
set_option Elab.async false

abbrev program := Solvers.P95Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

def items (limit : Nat) (known : Finset Nat) : List Mutable.Value :=
  (List.range (limit + 1)).map (fun n => .scalar (.bool (decide (n ∈ known))))

@[simp] theorem items_length (limit : Nat) (known : Finset Nat) : (items limit known).length = limit + 1 := by
  simp [items]

theorem items_lookup (limit n : Nat) (known : Finset Nat) (inside : n ≤ limit) :
    (items limit known)[n]? = some (.scalar (.bool (decide (n ∈ known)))) := by
  simp [items, show n < limit + 1 by omega]

theorem items_set (limit n : Nat) (known : Finset Nat) (inside : n ≤ limit) :
    (items limit known).set n (.scalar (.bool true)) = items limit (insert n known) := by
  apply List.ext_getElem?
  intro k
  by_cases bounded : k ≤ limit
  · by_cases same : n = k
    · subst k
      rw [List.getElem?_set_self (by simp; omega), items_lookup limit n _ inside]
      simp
    · rw [List.getElem?_set_ne same, items_lookup limit k _ bounded, items_lookup limit k _ bounded]
      simp [Ne.symm same]
  · simp only [List.getElem?_eq_none (by simp; omega : ((items limit known).set n (.scalar (.bool true))).length ≤ k),
      List.getElem?_eq_none (by simp; omega : (items limit (insert n known)).length ≤ k)]

variable {globals : Mutable.Globals}

theorem read (limit n address : Nat) (known : Finset Nat) (inside : n ≤ limit)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list (items limit known)))
    (getTable : env "processed" = some (.ref address)) (getIndex : env "n" = some (number n)) :
    Evaluates program env (.subscript (.name "processed") (.name "n")) heap
      (.scalar (.bool (decide (n ∈ known)))) heap globals := by
  apply Evaluates.subscript (by intros; simp) (Evaluates.lookup getTable) (Evaluates.lookup getIndex)
  have selected := items_lookup limit n known inside
  simpa only [(List.getElem?_eq_some_iff.mp selected).2] using
    Mutable.getItem_list heap address n _ found (by simp; omega)

abbrev body : List Stmt :=
  [.assign [.subscript (.name "processed") (.name "v")] (.constant (.bool true))]

theorem write (limit n address : Nat) (known : Finset Nat) (inside : n ≤ limit)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list (items limit known)))
    (getTable : env "processed" = some (.ref address)) (getIndex : env "v" = some (number n)) :
    BlockContinues program env body heap env
      (heap.set address (.list (items limit (insert n known)))) globals := by
  apply BlockContinues.setItem (by intros; simp) Evaluates.constant
    (.cons (Evaluates.lookup getTable) (.cons (Evaluates.lookup getIndex) .nil))
  simpa only [items_set limit n known inside] using
    HeapTables.setItem_list heap address n (items limit known) (.scalar (.bool true)) found (by simp; omega)

theorem loop (limit address pathAddress : Nat) (known : Finset Nat) (before remaining : List Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (different : address ≠ pathAddress)
    (inside : ∀ n ∈ remaining, n ≤ limit)
    (found : heap[address]? = some (.list (items limit known)))
    (pathFound : heap[pathAddress]? = some (.list ((before ++ remaining).map number)))
    (getTable : env "processed" = some (.ref address)) :
    ∃ finalEnv, BlockContinues program env
      [.sequenceStepLoop "v" (.ref pathAddress) before.length body []] heap finalEnv
      (heap.set address (.list (items limit (known ∪ remaining.toFinset)))) globals ∧
      ∀ name, name ≠ "v" → finalEnv name = env name := by
  induction remaining generalizing before heap env known with
  | nil =>
      have unchanged : heap.set address (.list (items limit known)) = heap := by
        obtain ⟨bounded, selected⟩ := List.getElem?_eq_some_iff.mp found
        rw [← selected, List.set_getElem_self bounded]
      simp only [List.toFinset_nil, Finset.union_empty, unchanged]
      refine ⟨env, ?_, fun _ _ => rfl⟩
      apply BlockContinues.sequence_done (items := before.map number)
      · simp [Mutable.sequence, sequenceValue, pathFound]
      · simp
      · exact .nil
  | cons n rest ih =>
      let entered := setLocal env "v" (number n)
      let updated := heap.set address (.list (items limit (insert n known)))
      have bounded := inside n (by simp)
      have enteredTable : entered "processed" = some (.ref address) := by
        simpa [entered, setLocal] using getTable
      have written := write (globals := globals) limit n address known bounded heap entered found enteredTable
        (by simp [entered, setLocal])
      have nextFound : updated[address]? = some (.list (items limit (insert n known))) :=
        List.getElem?_set_self (List.getElem?_eq_some_iff.mp found).1
      have nextPath : updated[pathAddress]? = some (.list (((before ++ [n]) ++ rest).map number)) := by
        simpa only [updated, List.getElem?_set_ne different, List.append_assoc, List.singleton_append] using pathFound
      obtain ⟨finalEnv, continued, stable⟩ := ih (insert n known) (before ++ [n]) updated entered
        (by intro k member; exact inside k (by simp [member])) nextFound nextPath enteredTable
      refine ⟨finalEnv, ?_, ?_⟩
      · apply BlockContinues.sequence_step (items := (before ++ n :: rest).map number) (value := number n)
        · simp [Mutable.sequence, sequenceValue, pathFound]
        · simp
        have union : insert n known ∪ rest.toFinset = known ∪ (n :: rest).toFinset := by ext k; simp
        simpa only [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
          Bool.false_eq_true, if_false, List.length_append, List.length_singleton,
          updated, List.set_set, union] using written.append continued
      · intro name notV
        exact (stable name notV).trans (by simp [entered, setLocal, notV])

theorem mark (limit address pathAddress : Nat) (known : Finset Nat) (path : List Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (different : address ≠ pathAddress)
    (inside : ∀ n ∈ path, n ≤ limit)
    (found : heap[address]? = some (.list (items limit known)))
    (pathFound : heap[pathAddress]? = some (.list (path.map number)))
    (getTable : env "processed" = some (.ref address)) (getPath : env "path" = some (.ref pathAddress)) :
    ∃ finalEnv, BlockContinues program env [.forIn (.name "v") (.name "path") body []] heap finalEnv
      (heap.set address (.list (items limit (known ∪ path.toFinset)))) globals ∧
      ∀ name, name ≠ "v" → finalEnv name = env name := by
  obtain ⟨finalEnv, executed, stable⟩ := loop limit address pathAddress known [] path heap env different
    inside found (by simpa using pathFound) getTable
  refine ⟨finalEnv, ?_, stable⟩
  apply BlockContinues.for_sequence (by intros; simp) (Evaluates.lookup getPath)
    (items := path.map number)
  · simp [Mutable.sequence, pathFound]
  · exact executed

end solvers.«95».processed
