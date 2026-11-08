import solvers.«84.py.lean.d».LandingCommunity

namespace solvers.«84».resolve_landing.chance
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false
set_option maxRecDepth 2048

def display : Expr := .dict [
  (.tuple [.name "square", .constant (.bool false)], .constant (.int 96)),
  (pool.chanceJail, .constant (.int 16))]

def descriptions (square : Int) : List TupleDictionary.Description :=
  [(pool.pair square false, .int 96), (pool.pair 10 true, .int 16)]

def iterable : Expr := match (pool.branch 2)[(1 : Nat)]? with
  | some (.forIn _ expression _ _) => expression
  | _ => .constant .none

def destinations (square : Int) : List Int :=
  [0, 11, 24, 39, 5, CyclicStops.nextStop [5, 15, 25, 35] 5 square,
    CyclicStops.nextStop [5, 15, 25, 35] 5 square, CyclicStops.nextStop [12, 28] 12 square, square - 3]

theorem display_source : (pool.branch 2)[0]? = some (.assign [.name "weights"] display) := rfl

theorem iterable_source : iterable = .tuple [.constant (.int 0), .constant (.int 11), .constant (.int 24),
    .constant (.int 39), .constant (.int 5), .call (.name "next_railway") [.name "square"],
    .call (.name "next_railway") [.name "square"], .call (.name "next_utility") [.name "square"],
    .binOp (.name "square") .sub (.constant (.int 3))] := rfl

theorem dictionary_result (square : Int) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getSquare : env "square" = some (.scalar (.int square))) :
    HeapMemoCalls.EvaluatesRel program pool.spec env display heap
      (TupleDictionary.Post heap (descriptions square)) globals := by
  apply TupleDictionary.display pool.catalog program env _ _ _ _ _ (by decide)
  · exact pool.dynamic_pair (.name "square") square false env heap globals (HeapMemoCalls.Evaluates.lookup getSquare)
  · intro middle
    apply TupleDictionary.arguments_cons pool.catalog program env
    · exact TupleDictionary.row_result pool.catalog program env pool.chanceJail (pool.pair 10 true) (.int 16) middle globals
        (pool.pair_result pool.chanceJail 10 true pool.chance_jail_source pool.known_chance_jail env middle globals)
    · intro afterJail
      exact TupleDictionary.arguments_nil pool.catalog program env afterJail globals

/-- Both railway cards call the source helper, preserving their repeated destination. -/
theorem iterable_result (square : Int) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getSquare : env "square" = some (.scalar (.int square))) :
    HeapCalls.Evaluates program env iterable heap (.ref heap.length)
      (heap ++ [.tuple ((destinations square).map fun value => .scalar (.int value))]) globals := by
  rw [iterable_source]
  apply HeapCalls.Evaluates.tuple (nonempty := by simp [destinations])
  apply HeapCalls.Evaluates.Arguments.cons HeapCalls.Evaluates.constant
  apply HeapCalls.Evaluates.Arguments.cons HeapCalls.Evaluates.constant
  apply HeapCalls.Evaluates.Arguments.cons HeapCalls.Evaluates.constant
  apply HeapCalls.Evaluates.Arguments.cons HeapCalls.Evaluates.constant
  apply HeapCalls.Evaluates.Arguments.cons HeapCalls.Evaluates.constant
  apply HeapCalls.Evaluates.Arguments.cons
    (HeapCalls.Evaluates.call (.cons (HeapCalls.Evaluates.lookup getSquare) .nil) (next_railway.result square heap globals))
  apply HeapCalls.Evaluates.Arguments.cons
    (HeapCalls.Evaluates.call (.cons (HeapCalls.Evaluates.lookup getSquare) .nil) (next_railway.result square heap globals))
  apply HeapCalls.Evaluates.Arguments.cons
    (HeapCalls.Evaluates.call (.cons (HeapCalls.Evaluates.lookup getSquare) .nil) (next_utility.result square heap globals))
  apply HeapCalls.Evaluates.Arguments.cons (middle := heap)
  · apply HeapCalls.Evaluates.binary (HeapCalls.Evaluates.lookup getSquare) HeapCalls.Evaluates.constant
    simp [Mutable.binary, scalarBinary]
  · exact .nil

end solvers.«84».resolve_landing.chance
