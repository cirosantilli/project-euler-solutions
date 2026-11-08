import solvers.«84.py.lean.d».LandingChanceInitial
import solvers.«84.py.lean.d».LandingMoveLoop
import solvers.«84.py.lean.d».LandingResult
import lang.py.FormalSW.HeapMemoLoopRules

namespace solvers.«84».resolve_landing.chance
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

def initialData (square : Int) : List weights.CardData := [((square, false), 96), ((10, true), 16)]
def data (square : Int) : List weights.CardData := initialData square ++ (destinations square).flatMap weights.moveData

theorem lookup_initial (square : Int) (stay jail : Nat) (landing : Int × Bool) :
    (weights.lookup (initialEntries square stay jail) landing).getD 0 = weights.total (initialData square) landing := by
  by_cases first : landing = (square, false)
  · subst landing
    simp [initialEntries, weights.lookup, weights.total, initialData]
  · by_cases second : landing = (10, true)
    · subst landing
      simp [initialEntries, weights.lookup, weights.total, initialData]
    · simp [initialEntries, weights.lookup, weights.total, initialData, first, second, eq_comm]

abbrev Post (base : Mutable.Heap) (square : Int) := weights.Post base (data square)

theorem Post.iteration {base : Mutable.Heap} {square : Int} {value : Mutable.Value} {after : Mutable.Heap}
    (post : Post base square value after) :
    ∃ root entries, value = .ref root ∧ after[root]? = some (.keyDict (weights.encode entries)) ∧
      weights.Valid after entries ∧ ∀ landing,
        (entries.map fun entry => if landing = entry.landing then entry.weight else 0).sum = weights.total (data square) landing :=
  weights.Post.iteration post

theorem Post.prefix {base extra : Mutable.Heap} {square : Int} {value : Mutable.Value} {after : Mutable.Heap}
    (post : Post (base ++ extra) square value after) : Post base square value after := weights.Post.prefix post

/-- The complete Chance branch aggregates all nine movement cards, including repeated destinations. -/
theorem branch_result (square : Int) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getSquare : env "square" = some (.scalar (.int square))) (following : List Stmt := []) :
    HeapMemoCalls.BlockReturnsRel program pool.spec env (pool.branch 2 ++ following) heap (Post heap square) globals := by
  apply HeapMemoCalls.BlockReturnsRel.assign (initial_result square env heap globals getSquare)
  rintro value after ⟨root, stay, jail, initial, rfl, rfl, fresh, found, valid, distinct, positive⟩
  let localEnv := HeapCalls.setLocal env "weights" (.ref root)
  have localSquare : localEnv "square" = some (.scalar (.int square)) := by simpa [localEnv, HeapCalls.setLocal] using getSquare
  let table := Mutable.Object.tuple ((destinations square).map fun value => .scalar (.int value))
  let middle := (heap ++ initial) ++ [table]
  apply HeapMemoCalls.BlockReturnsRel.for_sequence_heap pool.disjoint
    (items := (destinations square).map fun value => .scalar (.int value))
    (by intro expressions; rw [iterable_source]; simp)
    (iterable_result square localEnv (heap ++ initial) globals localSquare)
    (TuplePool.preserves_append pool.catalog (heap ++ initial) [table])
    (by simp [Mutable.sequence])
  have extended : middle[root]? = some (.keyDict (weights.encode (initialEntries square stay jail))) :=
    (List.getElem?_append_left (l₂ := [table]) (List.getElem?_eq_some_iff.mp found).1).trans found
  apply weights.move_loop (destinations square) [] localEnv middle globals (heap ++ initial).length root (initialEntries square stay jail)
    (by simp [middle, table]) extended (valid.append [table]) distinct positive (by simp [localEnv, HeapCalls.setLocal])
  intro finalEnv entries extra getWeights finalValid finalDistinct finalPositive counts
  apply HeapMemoCalls.BlockReturnsRel.return_value
  apply (HeapMemoCalls.Evaluates.lookup getWeights).rel.consequence
  rintro value after ⟨rfl, same⟩
  subst after
  refine ⟨root, entries, (initial ++ [table] ++ extra).set (root - heap.length) (.keyDict (weights.encode entries)), rfl,
    ?_, fresh, ?_, finalValid, finalDistinct, finalPositive, ?_⟩
  · simpa only [middle, List.append_assoc] using
      List.set_append_right (s := heap) (t := initial ++ [table] ++ extra) root (.keyDict (weights.encode entries)) fresh
  · apply List.getElem?_set_self
    have inside := (List.getElem?_eq_some_iff.mp extended).1
    simp only [List.length_append]
    omega
  · intro landing
    rw [counts, lookup_initial]
    simp [data, weights.total, List.map_append, List.sum_append]

end solvers.«84».resolve_landing.chance
