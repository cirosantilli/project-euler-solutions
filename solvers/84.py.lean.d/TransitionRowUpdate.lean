import solvers.«84.py.lean.d».LandingPool
import ProjectEulerSolutions.Lib.NaturalDictionary

namespace solvers.«84».transition_row
open FormalSW.PyAst HeapCalls ProjectEulerSolutions NaturalDictionary
set_option Elab.async false

def add (entries : Entries) (target amount : Nat) : Entries :=
  insert entries target ((lookup entries target).getD 0 + amount)

theorem lookup_add (entries : Entries) (target amount query : Nat) :
    (lookup (add entries target amount) query).getD 0 =
      (lookup entries query).getD 0 + if query = target then amount else 0 := by
  simp only [add, lookup_insert]
  by_cases same : query = target <;> simp [same]

def increment (expression : Expr) : Stmt := .assign [.subscript (.name "row") (.name "target")]
  (.binOp (.call (.attribute (.name "row") "get") [.name "target", .constant (.int 0)]) .add expression)

theorem increment_result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (root target amount : Nat) (entries : Entries) (expression : Expr)
    (found : Found heap root entries) (getRow : env "row" = some (.ref root))
    (getTarget : env "target" = some (number target))
    (evaluated : Evaluates program env expression heap (number amount) heap globals) :
    BlockContinues program env [increment expression] heap env
      (heap.set root (.dict (encode (add entries target amount)))) globals := by
  have read : Evaluates program env
      (.call (.attribute (.name "row") "get") [.name "target", .constant (.int 0)]) heap
      (number ((lookup entries target).getD 0)) heap globals := by
    apply Evaluates.dictionary_get (by intro name same; cases same; simp [getRow])
      (Evaluates.lookup getRow) found (.cons (Evaluates.lookup getTarget) (.cons Evaluates.constant .nil))
    simp only [HeapCalls.primitive, Mutable.dictionaryGet_scalar heap root (encode entries) found,
      lookup_encode]
    cases lookup entries target <;> rfl
  apply insert_evaluates heap env "row" "target" root target ((lookup entries target).getD 0 + amount) entries _
    found getRow getTarget
  apply Evaluates.binary read evaluated
  simp [Mutable.binary, scalarBinary, number]

theorem tuple_frame (heap : Mutable.Heap) (root : Nat) (entries replacement : Entries)
    (found : Found heap root entries) (address : Nat) (items : List Mutable.Value)
    (tupleFound : heap[address]? = some (.tuple items)) :
    (heap.set root (.dict (encode replacement)))[address]? = some (.tuple items) := by
  have different : root ≠ address := by
    intro same
    change heap[root]? = some (.dict (encode entries)) at found
    rw [same, tupleFound] at found
    cases found
  exact (List.getElem?_set_ne different).trans tupleFound

theorem increment_continue (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (root target amount : Nat) (entries : Entries) (expression : Expr)
    (found : Found heap root entries) (getRow : env "row" = some (.ref root))
    (getTarget : env "target" = some (number target))
    (evaluated : Evaluates program env expression heap (number amount) heap globals)
    (body : List Stmt) (post : Mutable.Value → Mutable.Heap → Prop)
    (rest : HeapMemoCalls.BlockReturnsRel program resolve_landing.pool.spec env body
      (heap.set root (.dict (encode (add entries target amount)))) post globals) :
    HeapMemoCalls.BlockReturnsRel program resolve_landing.pool.spec env (increment expression :: body) heap post globals := by
  intro anchors stack
  have run := HeapMemoCalls.Runs.from_heap (HeapCalls.Extends.refl program) resolve_landing.pool.disjoint
    (increment_result env heap globals root target amount entries expression found getRow getTarget evaluated body anchors stack)
    (TuplePool.preserves_frame resolve_landing.pool.catalog _ _ (tuple_frame heap root entries _ found))
  apply run.bind
  rintro state rfl
  exact rest anchors stack

end solvers.«84».transition_row
