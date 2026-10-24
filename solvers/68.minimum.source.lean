import solvers.«68.text.source»
import ProjectEulerStatements.Lib.MagicRingIndexMinimumCalls
import ProjectEulerSolutions.Lib.HeapCollection

namespace solvers.«68»
open FormalSW.PyAst HeapCalls ProjectEulerStatements MagicRing MagicRingTextCalls
set_option Elab.async false

private abbrev solveCode : List Stmt :=
  executableBody ((findFunctionIn program.body "solve").getD ([], [])).2
abbrev combinationBody : List Stmt :=
  match (solveCode[6]? : Option Stmt) with
  | some (.forIn _ _ instructions _) => instructions
  | _ => []
abbrev innerPermutationBody : List Stmt :=
  match (combinationBody[5]? : Option Stmt) with
  | some (.forIn _ _ instructions _) => instructions
  | _ => []
abbrev minimumExpression : Expr :=
  match (innerPermutationBody[4]? : Option Stmt) with
  | some (.assign _ expression) => expression
  | _ => .constant .none
abbrev indexPoolExpression : Expr :=
  .listComp (.name "$formalsw.range.item")
    [("$formalsw.range.item", .call (.name "range") [.name "n"], [])]

theorem minimum_expression_source : minimumExpression = .extremumSequence false
    indexPoolExpression "index" (.subscript (.name "outer") (.name "index")) := rfl

theorem index_pool_result (n : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getN : env "n" = some (number n)) :
    ∃ extra, Evaluates program env indexPoolExpression heap (.ref heap.length)
      (heap ++ (.list (encode (List.range n)) :: extra)) := by
  obtain ⟨extra, pooled⟩ := ProjectEulerSolutions.HeapCollection.range_map (globals := [])
    program "$formalsw.range.item" (.name "$formalsw.range.item") number
    (fun heap env value found => ⟨[], by simpa using
      (Evaluates.lookup (module := program) (heap := heap) found)⟩)
    heap env [.name "n"] [number n] 0 n
    (.cons (Evaluates.lookup getN) .nil)
    (by
      change Except.ok (ε := ExceptionObject) ((0 : Int), (n : Int), (1 : Int)) =
        .ok (0, (0 : Int) + n, 1)
      simp only [Int.zero_add])
  exact ⟨extra, by simpa only [indexPoolExpression, encode, List.range_eq_range',
    ProjectEulerSolutions.HeapCollection.working] using pooled⟩

theorem minimum_result {n : Nat} (ring : Ring n) (positive : 0 < n)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (outerAddress : Nat)
    (getOuter : heap[outerAddress]? = some (.list (encode ring.outer)))
    (localOuter : env "outer" = some (.ref outerAddress))
    (getN : env "n" = some (number n)) :
    ∃ extra, Evaluates program env minimumExpression heap (number (minimumIndex ring.outer))
      (heap ++ extra) := by
  obtain ⟨garbage, pooled⟩ := index_pool_result n heap env getN
  let extra : Mutable.Heap := .list (encode (List.range n)) :: garbage
  have outerFound : (heap ++ extra)[outerAddress]? = some (.list (encode ring.outer)) := by
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp getOuter).1]
    exact getOuter
  have poolFound : (heap ++ extra)[heap.length]? =
      some (.list (encode (List.range n))) := by simp [extra]
  refine ⟨extra, ?_⟩
  rw [minimum_expression_source]
  apply Evaluates.extremum_sequence pooled
    (items := encode (List.range n))
  · simp [Mutable.sequence]
  · exact MagicRingIndexMinimumCalls.result program ring positive (heap ++ extra) env
      outerAddress heap.length outerFound poolFound localOuter

end solvers.«68»
