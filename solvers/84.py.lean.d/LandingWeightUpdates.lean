import solvers.«84.py.lean.d».LandingWeights

namespace solvers.«84».resolve_landing.weights
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

def increment (expression : Expr) : Stmt := .assign [.subscript (.name "weights") (.name "outcome")]
  (.binOp (.call (.attribute (.name "weights") "get") [.name "outcome", .constant (.int 0)]) .add expression)

theorem increment_result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (root : Nat) (entries : List Entry) (landing : Int × Bool) (address : Nat) (amount : Int) (expression : Expr)
    (found : heap[root]? = some (.keyDict (encode entries)))
    (getWeights : env "weights" = some (.ref root)) (getOutcome : env "outcome" = some (.ref address))
    (key : TuplePool.View heap (.tuple (pool.pair landing.1 landing.2)) (.ref address))
    (evaluated : HeapCalls.Evaluates program env expression heap (.scalar (.int amount)) heap globals) :
    HeapCalls.BlockContinues program env [increment expression] heap env
      (heap.set root (.keyDict (encode (add entries landing address amount)))) globals := by
  have read : HeapCalls.Evaluates program env
      (.call (.attribute (.name "weights") "get") [.name "outcome", .constant (.int 0)]) heap
      (.scalar (.int ((lookup entries landing).getD 0))) heap globals := by
    apply HeapCalls.Evaluates.dictionary_get_bound
      (by intro name same; cases same; simp [getWeights]) (HeapCalls.Evaluates.lookup getWeights)
      (by simp [HeapCalls.bindMethod, found])
      (.cons (HeapCalls.Evaluates.lookup getOutcome) (.cons HeapCalls.Evaluates.constant .nil))
    simp only [HeapCalls.primitive, Mutable.dictionaryGet_keyed heap root (encode entries) found _ _ key.hash,
      lookup_encode]
    cases lookup entries landing <;> rfl
  apply HeapCalls.BlockContinues.setItem (value := .scalar (.int ((lookup entries landing).getD 0 + amount)))
    (afterRhs := heap) (by intro a b c; simp)
    (HeapCalls.Evaluates.binary read evaluated (by simp [Mutable.binary, scalarBinary]))
    (.cons (HeapCalls.Evaluates.lookup getWeights) (.cons (HeapCalls.Evaluates.lookup getOutcome) .nil))
  simp [Mutable.setItem, found, key.hash, insert_encode, add, Except.map]

theorem increment_preserves (heap : Mutable.Heap) (root : Nat) (entries : List Entry)
    (landing : Int × Bool) (address : Nat) (amount : Int)
    (found : heap[root]? = some (.keyDict (encode entries))) :
    HeapMemoCalls.Cache.Preserves pool.spec heap
      (heap.set root (.keyDict (encode (add entries landing address amount)))) :=
  TuplePool.preserves_frame pool.catalog _ _ (TupleDictionary.tuple_frame heap root _ _ found)

theorem increment_valid (heap : Mutable.Heap) (root : Nat) (entries : List Entry)
    (landing : Int × Bool) (address : Nat) (amount : Int)
    (found : heap[root]? = some (.keyDict (encode entries))) (valid : Valid heap entries)
    (key : TuplePool.View heap (.tuple (pool.pair landing.1 landing.2)) (.ref address)) :
    Valid (heap.set root (.keyDict (encode (add entries landing address amount)))) (add entries landing address amount) := by
  intro entry member
  exact TuplePool.View.frame _ _ (TupleDictionary.tuple_frame heap root _ _ found) _ _
    (valid.inserted landing address _ key entry member)

/-- The body updates one private dictionary while preserving pooled keys for the continuation. -/
theorem increment_continue (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (root : Nat) (entries : List Entry) (landing : Int × Bool) (address : Nat) (amount : Int) (expression : Expr)
    (found : heap[root]? = some (.keyDict (encode entries)))
    (getWeights : env "weights" = some (.ref root)) (getOutcome : env "outcome" = some (.ref address))
    (key : TuplePool.View heap (.tuple (pool.pair landing.1 landing.2)) (.ref address))
    (evaluated : HeapCalls.Evaluates program env expression heap (.scalar (.int amount)) heap globals)
    (body : List Stmt) (post : Mutable.Value → Mutable.Heap → Prop)
    (rest : HeapMemoCalls.BlockReturnsRel program pool.spec env body
      (heap.set root (.keyDict (encode (add entries landing address amount)))) post globals) :
    HeapMemoCalls.BlockReturnsRel program pool.spec env (increment expression :: body) heap post globals := by
  intro anchors stack
  have run := HeapMemoCalls.Runs.from_heap (HeapCalls.Extends.refl program) pool.disjoint
    (increment_result env heap globals root entries landing address amount expression found getWeights getOutcome key evaluated body anchors stack)
    (increment_preserves heap root entries landing address amount found)
  apply run.bind
  rintro state rfl
  exact rest anchors stack

end solvers.«84».resolve_landing.weights
