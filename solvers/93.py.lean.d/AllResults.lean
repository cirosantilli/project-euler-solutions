import solvers.«93.py.lean.d».InitialState
import solvers.«93.py.lean.d».Recursive
import ProjectEulerSolutions.Lib.HeapFramePrefix

namespace solvers.«93».all_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions
set_option Elab.async false

abbrev args : List Arg := ((findFunctionIn program.body "all_results").getD ([], [])).1
abbrev body : List Stmt := ((findFunctionIn program.body "all_results").getD ([], [])).2

theorem source_body : executableBody body =
    [.assign [.name "start"] startExpr, .assign [.name cellName] (.list [.dict []]),
      .return (.call (.name recName) [.name "start", .name cellName])] := rfl

/-- The full source call returns a fresh complete result set and preserves every caller object. -/
def Verified (digits : List Int) (address : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals) : Prop :=
  ∃ result extra, heap.length ≤ result ∧
    ReturnsValue program "all_results" [.ref address] heap (.ref result) (heap ++ extra) globals ∧
    Heap.ResultAt (heap ++ extra) result (integerPairs digits)

theorem result (digits : List Int) (address : Nat) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (found : heap[address]? = some (.tuple (digits.map Heap.number))) : Verified digits address heap globals := by
  let env := functionEnv args body [.ref address] globals
  have getDigits : env "digits" = some (.ref address) := by cases globals <;> rfl
  obtain ⟨extra, state, items, pairs, prepared, stateFound, rows, permutation, canonical⟩ :=
    initial_state heap address digits env globals found getDigits
  let middle := heap ++ extra
  let ready := middle ++ [Mutable.Object.dict [], .list [.ref middle.length]]
  let started := setLocal env "start" (.ref state)
  let initialized := setLocal started cellName (.ref (middle.length + 1))
  have getState : initialized "start" = some (.ref state) := by simp [initialized, started, setLocal, cellName]
  have getCell : initialized cellName = some (.ref (middle.length + 1)) := by simp [initialized, setLocal]
  have cellFound : ready[middle.length + 1]? = some (.list [.ref middle.length]) := by simp [ready]
  have memoFound : ready[middle.length]? = some (.dict []) := by simp [ready]
  have valid : MemoValid ready middle.length [] := MemoValid.empty ready middle.length memoFound
  have readyState := HeapCollection.prefix_found middle [.dict [], .list [.ref middle.length]] state _ stateFound
  have readyRows := rows.append [.dict [], .list [.ref middle.length]]
  obtain ⟨after, result, entries, returned, post⟩ :=
    recursive_verified ready (middle.length + 1) middle.length state items pairs [] globals cellFound valid readyState readyRows canonical
  have frame : PreservesExcept middle.length heap after :=
    ((PreservesExcept.append middle.length heap extra).trans
      (PreservesExcept.append middle.length middle [.dict [], .list [.ref middle.length]])).trans post.frame
  obtain ⟨finalExtra, same⟩ := frame.prefix_of_fresh (by simp [middle])
  have complete : Heap.ResultAt after result (integerPairs digits) := by
    obtain ⟨collected, set, members⟩ := post.complete
    exact ⟨collected, set, fun pair => (members pair).trans (expected_perm permutation pair)⟩
  have fresh : heap.length ≤ result := by
    have origin := post.resultOrigin
    simp only [List.map_nil, List.not_mem_nil, false_or] at origin
    simp only [ready, middle, List.length_append, List.length_cons, List.length_nil] at origin
    omega
  refine ⟨result, finalExtra, fresh, ?_, same ▸ complete⟩
  rw [same]
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign prepared
  have allocated : Evaluates program started (.list [.dict []]) middle (.ref (middle.length + 1)) ready globals := by
    simpa only [ready, List.length_append, List.length_singleton, List.append_assoc, List.singleton_append] using
      (Evaluates.list (module := program) (env := started) (globals := globals) (.cons (Evaluates.dictionary .nil) .nil) (heap := middle))
  apply BlockReturns.assign allocated
  exact BlockReturns.return_value (Evaluates.call (.cons (Evaluates.lookup getState) (.cons (Evaluates.lookup getCell) .nil)) returned)

end solvers.«93».all_results
