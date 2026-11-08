import solvers.«84.py.lean.d».LandingJail
import ProjectEulerSolutions.Lib.TupleDictionaryDisplay

namespace solvers.«84».resolve_landing.community
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false
set_option maxRecDepth 2048

def display : Expr := .dict [
  (.tuple [.name "square", .constant (.bool false)], .constant (.int 224)),
  (pool.communityGo, .constant (.int 16)), (pool.communityJail, .constant (.int 16))]

def descriptions (square : Int) : List TupleDictionary.Description :=
  [(pool.pair square false, .int 224), (pool.pair 0 false, .int 16), (pool.pair 10 true, .int 16)]

theorem source : pool.branch 1 = [.return display] := rfl

/-- Community Chest retains the square for fourteen cards and moves to GO or JAIL for one each. -/
theorem dictionary_result (square : Int) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getSquare : env "square" = some (.scalar (.int square))) :
    HeapMemoCalls.EvaluatesRel program pool.spec env display heap
      (TupleDictionary.Post heap (descriptions square)) globals := by
  apply TupleDictionary.display pool.catalog program env _ _ _ _ _ (by decide)
  · exact pool.dynamic_pair (.name "square") square false env heap globals (HeapMemoCalls.Evaluates.lookup getSquare)
  · intro middle
    apply TupleDictionary.arguments_cons pool.catalog program env
    · exact TupleDictionary.row_result pool.catalog program env pool.communityGo (pool.pair 0 false) (.int 16) middle globals
        (pool.pair_result pool.communityGo 0 false pool.community_go_source pool.known_community_go env middle globals)
    · intro afterGo
      apply TupleDictionary.arguments_cons pool.catalog program env
      · exact TupleDictionary.row_result pool.catalog program env pool.communityJail (pool.pair 10 true) (.int 16) afterGo globals
          (pool.pair_result pool.communityJail 10 true pool.community_jail_source pool.known_community_jail env afterGo globals)
      · intro afterJail
        exact TupleDictionary.arguments_nil pool.catalog program env afterJail globals

theorem result (square : Int) (member : square ∈ [2, 17, 33])
    (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.ReturnsRel program pool.spec "resolve_landing" [.scalar (.int square)] heap
      (TupleDictionary.Post heap (descriptions square)) globals := by
  let env := HeapCalls.functionEnv pool.definition.1 pool.definition.2 [.scalar (.int square)] globals
  have getSquare : env "square" = some (.scalar (.int square)) := by cases globals <;> rfl
  have notJail : square ≠ 30 := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    omega
  have jailTest : HeapMemoCalls.Evaluates program pool.spec env (.compare (.name "square") [(.eq, .constant (.int 30))])
      heap (.scalar (.bool false)) heap globals := by
    apply HeapMemoCalls.Evaluates.compare (HeapMemoCalls.Evaluates.lookup getSquare) HeapMemoCalls.Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, notJail]
  let temporary := Mutable.Object.tuple [.scalar (.int 2), .scalar (.int 17), .scalar (.int 33)]
  have communityTest : HeapCalls.Evaluates program env
      (.membership (.name "square") (.tuple [.constant (.int 2), .constant (.int 17), .constant (.int 33)]) false)
      heap (.scalar (.bool true)) (heap ++ [temporary]) globals := by
    apply HeapCalls.Evaluates.membership (.cons (HeapCalls.Evaluates.lookup getSquare) (.cons
      (HeapCalls.Evaluates.tuple (.cons HeapCalls.Evaluates.constant (.cons HeapCalls.Evaluates.constant
        (.cons HeapCalls.Evaluates.constant .nil))) (by simp)) .nil))
    simpa [member] using HeapCalls.membership_tuple_ints (heap ++ [temporary]) heap.length square [2, 17, 33] (by simp [temporary]) false
  have cachedTest := HeapMemoCalls.Evaluates.from_heap (HeapCalls.Extends.refl program) pool.disjoint communityTest
    (TuplePool.preserves_append pool.catalog heap [temporary])
  apply HeapMemoCalls.BlockReturnsRel.function (args := pool.definition.1) (body := pool.definition.2) (by decide) rfl rfl
  apply HeapMemoCalls.BlockReturnsRel.branch jailTest.rel
  rintro value after ⟨rfl, same⟩
  subst after
  refine ⟨false, rfl, ?_⟩
  apply HeapMemoCalls.BlockReturnsRel.branch cachedTest.rel
  rintro value after ⟨rfl, same⟩
  subst after
  refine ⟨true, rfl, ?_⟩
  apply HeapMemoCalls.BlockReturnsRel.return_value
  apply (dictionary_result square env (heap ++ [temporary]) globals getSquare).consequence
  exact fun _ _ post => post.prefix

end solvers.«84».resolve_landing.community
