import solvers.«84.py.lean.d».TransitionSource
import solvers.«84.py.lean.d».Encoding

namespace solvers.«84».transition_row
open FormalSW.PyAst HeapCalls ProjectEulerSolutions NaturalDictionary
set_option Elab.async false

theorem jail_body (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (root count : Nat) (entries : Entries) (found : Found heap root entries)
    (getRow : env "row" = some (.ref root)) (getCount : env "count" = some (number count)) :
    BlockContinues program env jailCode heap (setLocal env "target" (number 30))
      (heap.set root (.dict (encode (add entries 30 (256 * count))))) globals := by
  have target : Evaluates program env (.call (.name "encode_state") [.constant (.int 10), .constant (.int 0)])
      heap (number 30) heap globals :=
    Evaluates.call (.cons Evaluates.constant (.cons Evaluates.constant .nil)) (encode_state.natural 10 0 heap globals)
  have amount : Evaluates program (setLocal env "target" (number 30))
      (.binOp (.constant (.int 256)) .mult (.name "count")) heap (number (256 * count)) heap globals := by
    apply Evaluates.binary Evaluates.constant (Evaluates.lookup (by simpa [setLocal] using getCount))
    simp [Mutable.binary, scalarBinary, number]
  have updated := increment_result (setLocal env "target" (number 30)) heap globals root 30 (256 * count) entries _ found
    (by simpa [setLocal] using getRow) (by simp [setLocal]) amount
  rw [jail_source]
  exact (BlockContinues.assign target).append updated

/-- The source's third consecutive double adds the full roll weight to jail and skips card handling. -/
theorem jail_result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (root count : Nat) (entries : Entries) (found : Found heap root entries)
    (getRow : env "row" = some (.ref root)) (getCount : env "count" = some (number count))
    (getDouble : env "is_double" = some (.scalar (.bool true))) (getDoubles : env "doubles" = some (number 2)) :
    BlockContinues program env code heap (setLocal env "target" (number 30))
      (heap.set root (.dict (encode (add entries 30 (256 * count))))) globals := by
  have tested : Evaluates program env
      (.boolOp .and [.name "is_double", .compare (.name "doubles") [(.eq, .constant (.int 2))]]) heap
      (.scalar (.bool true)) heap globals := by
    apply Evaluates.boolean_next (Evaluates.lookup getDouble) rfl rfl
    apply Evaluates.boolean_last (flag := true) _ rfl
    apply Evaluates.compare (Evaluates.lookup getDoubles) Evaluates.constant
    rfl
  rw [code_source]
  exact BlockContinues.branch (flag := true) tested rfl (jail_body env heap globals root count entries found getRow getCount)

theorem jail_continue (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (root count : Nat) (entries : Entries) (found : Found heap root entries)
    (getRow : env "row" = some (.ref root)) (getCount : env "count" = some (number count))
    (getDouble : env "is_double" = some (.scalar (.bool true))) (getDoubles : env "doubles" = some (number 2))
    (body : List Stmt) (post : Mutable.Value → Mutable.Heap → Prop)
    (rest : HeapMemoCalls.BlockReturnsRel program resolve_landing.pool.spec (setLocal env "target" (number 30)) body
      (heap.set root (.dict (encode (add entries 30 (256 * count))))) post globals) :
    HeapMemoCalls.BlockReturnsRel program resolve_landing.pool.spec env (code ++ body) heap post globals := by
  intro anchors stack
  have run := HeapMemoCalls.Runs.from_heap (HeapCalls.Extends.refl program) resolve_landing.pool.disjoint
    (jail_result env heap globals root count entries found getRow getCount getDouble getDoubles body anchors stack)
    (TuplePool.preserves_frame resolve_landing.pool.catalog _ _ (tuple_frame heap root entries _ found))
  apply run.bind
  rintro state rfl
  exact rest anchors stack

end solvers.«84».transition_row
