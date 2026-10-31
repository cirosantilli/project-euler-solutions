import solvers.«93.py.lean.d».PairOperations
import ProjectEulerSolutions.Lib.RationalKeys
import lang.py.FormalSW.MembershipCalls
import lang.py.FormalSW.KeyDictionaryCalls

namespace solvers.«93».all_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
set_option Elab.async false

abbrev program := Solvers.P93Ast.module
abbrev recName := "__formalsw_local_all_results_rec"
abbrev cellName := "__formalsw_cell_all_results_memo"
abbrev memoExpr : Expr := .subscript (.name cellName) (.constant (.int 0))
abbrev cachedExpr : Expr := .subscript memoExpr (.name "state")
abbrev cachedGuard : Expr := .membership (.name "state") memoExpr false
abbrev recArgs : List Arg := ((findFunctionIn program.body recName).getD ([], [])).1
abbrev recBody : List Stmt := ((findFunctionIn program.body recName).getD ([], [])).2

theorem source_cached : recBody[0]? = some (.ifThen cachedGuard [.return cachedExpr] []) := rfl

theorem read_memo (heap : Mutable.Heap) (cell memo : Nat) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : heap[cell]? = some (.list [.ref memo])) (getCell : env cellName = some (.ref cell)) :
    Evaluates program env memoExpr heap (.ref memo) heap globals := by
  apply Evaluates.subscript (by intro a b c impossible; cases impossible) (Evaluates.lookup getCell) Evaluates.constant
  exact Mutable.getItem_list heap cell 0 [.ref memo] found (by simp)

abbrev MemoAt (heap : Mutable.Heap) (address : Nat) (entries : List (HashDict.Entry Mutable.Value)) : Prop :=
  (heap[address]?).bind Mutable.dictionaryEntries = some entries

theorem MemoAt.bound {heap : Mutable.Heap} {address : Nat} {entries : List (HashDict.Entry Mutable.Value)}
    (found : MemoAt heap address entries) : address < heap.length := by
  cases present : heap[address]? with
  | none => simp [MemoAt, present] at found
  | some object => exact (List.getElem?_eq_some_iff.mp present).1

theorem MemoAt.append {heap : Mutable.Heap} {address : Nat} {entries : List (HashDict.Entry Mutable.Value)}
    (found : MemoAt heap address entries) (extra : Mutable.Heap) : MemoAt (heap ++ extra) address entries := by
  unfold MemoAt
  rw [List.getElem?_append_left found.bound]
  exact found

/-- Memo writes cannot alias a cell whose object type is not a dictionary. -/
theorem MemoAt.other {heap : Mutable.Heap} {memo : Nat} {entries : List (HashDict.Entry Mutable.Value)}
    (found : MemoAt heap memo entries) (address : Nat) (object : Mutable.Object)
    (present : heap[address]? = some object) (ordinary : Mutable.dictionaryEntries object = none)
    (replacement : List (HashDict.Entry Mutable.Value)) :
    (heap.set memo (.keyDict replacement))[address]? = some object := by
  have different : memo ≠ address := by
    intro equal
    subst address
    simp [MemoAt, present, ordinary] at found
  simpa [List.getElem?_set, different, Ne.symm different] using present

private theorem lookup_insert_self (entries : List (HashDict.Entry Mutable.Value))
    (key : HashKey) (original value : Mutable.Value) (reflexive : HashKey.equal key key = true) :
    HashDict.lookup (HashDict.insert entries key original value) key = some value := by
  induction entries with
  | nil => simp [HashDict.insert, HashDict.lookup_cons, reflexive]
  | cons entry rest ih =>
    cases same : HashKey.equal entry.1 key <;> simp [HashDict.insert, HashDict.lookup_cons, same, ih]

theorem stored_lookup (entries : List (HashDict.Entry Mutable.Value)) (pairs : List Pair)
    (state : Nat) (value : Mutable.Value) :
    HashDict.lookup (HashDict.insert entries (Heap.stateKey pairs) (.ref state) value) (Heap.stateKey pairs) = some value :=
  lookup_insert_self entries _ _ _ (by simp [Heap.stateKey_equal])

private theorem read_cache (heap : Mutable.Heap) (memo state : Nat) (entries : List (HashDict.Entry Mutable.Value))
    (hashed : HashKey) (cached : Mutable.Value) (found : MemoAt heap memo entries)
    (hash : Mutable.hashKey (.ref state) heap = some (.ok hashed, heap))
    (lookup : HashDict.lookup entries hashed = some cached) :
    Mutable.getItem (.ref memo) (.ref state) heap = some (.ok cached, heap) := by
  cases stored : heap[memo]? with
  | none => simp [MemoAt, stored] at found
  | some object =>
    cases object <;> simp [MemoAt, stored, Mutable.dictionaryEntries] at found
    all_goals cases found; simp [Mutable.getItem, stored, hash, lookup]

private theorem test_cache (heap : Mutable.Heap) (memo state : Nat) (entries : List (HashDict.Entry Mutable.Value))
    (hashed : HashKey) (found : MemoAt heap memo entries)
    (hash : Mutable.hashKey (.ref state) heap = some (.ok hashed, heap)) :
    membership (.ref state) (.ref memo) false heap =
      some (.ok (.scalar (.bool (HashDict.lookup entries hashed).isSome)), heap) := by
  cases stored : heap[memo]? with
  | none => simp [MemoAt, stored] at found
  | some object =>
    cases object <;> simp [MemoAt, stored, Mutable.dictionaryEntries] at found
    all_goals cases found; simp [membership, stored, hash, Except.map]

theorem write_cache (heap : Mutable.Heap) (memo state : Nat) (entries : List (HashDict.Entry Mutable.Value))
    (hashed : HashKey) (cached : Mutable.Value) (found : MemoAt heap memo entries)
    (hash : Mutable.hashKey (.ref state) heap = some (.ok hashed, heap)) :
    Mutable.setItem (.ref memo) (.ref state) cached heap =
      some (.ok (.scalar .none), heap.set memo (.keyDict (HashDict.insert entries hashed (.ref state) cached))) := by
  cases stored : heap[memo]? with
  | none => simp [MemoAt, stored] at found
  | some object =>
    cases object <;> simp [MemoAt, stored, Mutable.dictionaryEntries] at found
    all_goals cases found; simp [Mutable.setItem, stored, hash, Except.map]

theorem read_cached (heap : Mutable.Heap) (cell memo state : Nat) (entries : List (HashDict.Entry Mutable.Value))
    (hashed : HashKey) (cached : Mutable.Value) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (cellFound : heap[cell]? = some (.list [.ref memo])) (memoFound : MemoAt heap memo entries)
    (hash : Mutable.hashKey (.ref state) heap = some (.ok hashed, heap))
    (lookup : HashDict.lookup entries hashed = some cached)
    (getCell : env cellName = some (.ref cell)) (getState : env "state" = some (.ref state)) :
    Evaluates program env cachedExpr heap cached heap globals := by
  apply Evaluates.subscript (by intro a b c impossible; cases impossible) (read_memo heap cell memo env globals cellFound getCell) (Evaluates.lookup getState)
  exact read_cache heap memo state entries hashed cached memoFound hash lookup

theorem cached_test (heap : Mutable.Heap) (cell memo state : Nat) (entries : List (HashDict.Entry Mutable.Value))
    (hashed : HashKey) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (cellFound : heap[cell]? = some (.list [.ref memo])) (memoFound : MemoAt heap memo entries)
    (hash : Mutable.hashKey (.ref state) heap = some (.ok hashed, heap))
    (getCell : env cellName = some (.ref cell)) (getState : env "state" = some (.ref state)) :
    Evaluates program env cachedGuard heap (.scalar (.bool (HashDict.lookup entries hashed).isSome)) heap globals := by
  apply Evaluates.membership (.cons (Evaluates.lookup getState)
    (.cons (read_memo heap cell memo env globals cellFound getCell) .nil))
  exact test_cache heap memo state entries hashed memoFound hash

/-- A memo hit returns the original stored object and leaves every caller object unchanged. -/
theorem cache_hit (heap : Mutable.Heap) (cell memo state : Nat) (entries : List (HashDict.Entry Mutable.Value))
    (values : List Mutable.Value) (pairs : List Pair) (cached : Mutable.Value) (globals : Mutable.Globals)
    (cellFound : heap[cell]? = some (.list [.ref memo])) (memoFound : MemoAt heap memo entries)
    (stateFound : heap[state]? = some (.tuple values)) (rows : Heap.Rows heap values pairs)
    (lookup : HashDict.lookup entries (Heap.stateKey pairs) = some cached) :
    ReturnsValue program recName [.ref state, .ref cell] heap cached heap globals := by
  let env := functionEnv recArgs recBody [.ref state, .ref cell] globals
  have getCell : env cellName = some (.ref cell) := by cases globals <;> rfl
  have getState : env "state" = some (.ref state) := by cases globals <;> rfl
  have hash := Heap.state_hash heap state values pairs stateFound rows
  have tested := cached_test heap cell memo state entries (Heap.stateKey pairs) env globals
    cellFound memoFound hash getCell getState
  have returned := read_cached heap cell memo state entries (Heap.stateKey pairs) cached env globals
    cellFound memoFound hash lookup getCell getState
  apply BlockReturns.function rfl rfl
  apply BlockReturns.branch (flag := true) tested (by simp [lookup, Mutable.truth, scalarTruth])
  exact BlockReturns.return_value returned

/-- Cache stores evaluate the result expression before reading the shared memo cell and key. -/
theorem store_result (heap middle : Mutable.Heap) (cell memo state : Nat)
    (entries : List (HashDict.Entry Mutable.Value)) (hashed : HashKey) (cached : Mutable.Value)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (expression : Expr)
    (rhs : Evaluates program env expression heap cached middle globals)
    (cellFound : middle[cell]? = some (.list [.ref memo])) (memoFound : MemoAt middle memo entries)
    (hash : Mutable.hashKey (.ref state) middle = some (.ok hashed, middle))
    (getCell : env cellName = some (.ref cell)) (getState : env "state" = some (.ref state)) :
    BlockContinues program env [.assign [.subscript memoExpr (.name "state")] expression] heap env
      (middle.set memo (.keyDict (HashDict.insert entries hashed (.ref state) cached))) globals := by
  apply BlockContinues.setItem (by intro a b c impossible; cases impossible) rhs
    (.cons (read_memo middle cell memo env globals cellFound getCell) (.cons (Evaluates.lookup getState) .nil))
  exact write_cache middle memo state entries hashed cached memoFound hash

end solvers.«93».all_results
