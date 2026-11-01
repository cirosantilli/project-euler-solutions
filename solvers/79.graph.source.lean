import «.formalsw».solvers.«79.py.ast»
import lang.py.FormalSW.DictionaryCalls

/-! Source-bound graph updates retain shared set references and count each edge once. -/
namespace solvers.«79».graph
open FormalSW.PyAst HeapCalls
set_option Elab.async false

abbrev program := Solvers.P79Ast.module
abbrev text (value : String) : Mutable.Value := .scalar (.string value)
abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
abbrev stringSet (values : List String) : List Constant := values.map Constant.string

theorem contains_strings (values : List String) (value : String) :
    ScalarSet.contains (stringSet values) (.string value) = decide (value ∈ values) := by
  apply Bool.eq_iff_iff.mpr
  simp [ScalarSet.contains, stringSet, List.any_eq_true, scalarCompare]

def addEdgeBody : List Stmt :=
  [.ifThen (.membership (.name "v") (.subscript (.name "adj") (.name "u")) true)
    [.expr (.call (.attribute (.subscript (.name "adj") (.name "u")) "add") [.name "v"]),
     .augAssign (.subscript (.name "indeg") (.name "v")) .add (.constant (.int 1))] []]

theorem addEdge_source :
    executableBody ((findFunctionIn program.body "__formalsw_local_build_graph_add_edge").getD
      ([], [])).2 = addEdgeBody := rfl

def edgeHeap (heap : Mutable.Heap) (neighborsAddress degreesAddress : Nat)
    (neighbors : List String) (degrees : List (Constant × Mutable.Value))
    (target : String) (count : Int) : Mutable.Heap :=
  if target ∈ neighbors then heap else
    (heap.set neighborsAddress (.set (stringSet (neighbors ++ [target])))).set degreesAddress
      (.dict (ScalarDict.insert degrees (.string target) (number (count + 1))))

theorem addEdge_block (env : HeapCalls.Env) (heap : Mutable.Heap)
    (source target : String) (adjacencyAddress degreesAddress neighborsAddress : Nat)
    (adjacency degrees : List (Constant × Mutable.Value)) (neighbors : List String) (count : Int)
    (getSource : env "u" = some (text source)) (getTarget : env "v" = some (text target))
    (getAdjacency : env "adj" = some (.ref adjacencyAddress))
    (getDegrees : env "indeg" = some (.ref degreesAddress))
    (adjacencyFound : heap[adjacencyAddress]? = some (.dict adjacency))
    (neighborsFound : heap[neighborsAddress]? = some (.set (stringSet neighbors)))
    (degreesFound : heap[degreesAddress]? = some (.dict degrees))
    (neighborsLookup : ScalarDict.lookup adjacency (.string source) = some (.ref neighborsAddress))
    (countLookup : ScalarDict.lookup degrees (.string target) = some (number count)) :
    BlockContinues program env addEdgeBody heap env
      (edgeHeap heap neighborsAddress degreesAddress neighbors degrees target count) := by
  have readNeighbors : Evaluates program env (.subscript (.name "adj") (.name "u"))
      heap (.ref neighborsAddress) heap := by
    apply Evaluates.subscript (by intros; intro impossible; cases impossible)
      (Evaluates.lookup getAdjacency) (Evaluates.lookup getSource)
    exact Mutable.getItem_dictionary heap adjacencyAddress adjacency adjacencyFound
      (.string source) neighborsLookup
  have tested : Evaluates program env
      (.membership (.name "v") (.subscript (.name "adj") (.name "u")) true) heap
      (.scalar (.bool (decide (target ∉ neighbors)))) heap := by
    apply Evaluates.membership (.cons (Evaluates.lookup getTarget) (.cons readNeighbors .nil))
    simp [HeapCalls.membership, neighborsFound, contains_strings]
  by_cases present : target ∈ neighbors
  · unfold edgeHeap
    rw [if_pos present]
    apply BlockContinues.branch (flag := false)
      (by simpa [present] using tested) rfl
    exact .nil
  · have distinct : neighborsAddress ≠ degreesAddress := by
      intro same
      subst degreesAddress
      rw [neighborsFound] at degreesFound
      cases degreesFound
    let added := heap.set neighborsAddress (.set (stringSet (neighbors ++ [target])))
    have add : BlockContinues program env
        [.expr (.call (.attribute (.subscript (.name "adj") (.name "u")) "add") [.name "v"])]
        heap env added := by
      apply BlockContinues.discard
      have operation := Evaluates.add readNeighbors neighborsFound (Evaluates.lookup getTarget)
        neighborsFound
      simpa [ScalarSet.insert, contains_strings, present, added, stringSet] using operation
    have degreesAfter : added[degreesAddress]? = some (.dict degrees) := by
      simp [added, distinct, degreesFound]
    have increment : BlockContinues program env
        [.augAssign (.subscript (.name "indeg") (.name "v")) .add (.constant (.int 1))]
        added env (added.set degreesAddress
          (.dict (ScalarDict.insert degrees (.string target) (number (count + 1))))) := by
      apply BlockContinues.augItem (updated := number (count + 1)) (afterOperation := added)
        (by intros; intro impossible; cases impossible)
        (Evaluates.lookup getDegrees) (Evaluates.lookup getTarget)
        (Mutable.getItem_dictionary added degreesAddress degrees degreesAfter
          (.string target) countLookup) Evaluates.constant
      · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
      · exact Mutable.setItem_dictionary added degreesAddress degrees degreesAfter
          (.string target) (number (count + 1))
    unfold edgeHeap
    rw [if_neg present]
    apply BlockContinues.branch (flag := true)
      (by simpa [present] using tested) rfl
    exact add.append increment

/-- The lifted source helper skips duplicate edges and increments exactly one destination count. -/
theorem addEdge_result (heap : Mutable.Heap)
    (source target : String) (adjacencyAddress degreesAddress neighborsAddress : Nat)
    (adjacency degrees : List (Constant × Mutable.Value)) (neighbors : List String) (count : Int)
    (adjacencyFound : heap[adjacencyAddress]? = some (.dict adjacency))
    (neighborsFound : heap[neighborsAddress]? = some (.set (stringSet neighbors)))
    (degreesFound : heap[degreesAddress]? = some (.dict degrees))
    (neighborsLookup : ScalarDict.lookup adjacency (.string source) = some (.ref neighborsAddress))
    (countLookup : ScalarDict.lookup degrees (.string target) = some (number count)) :
    ReturnsValue program "__formalsw_local_build_graph_add_edge"
      [text source, text target, .ref adjacencyAddress, .ref degreesAddress] heap (.scalar .none)
      (edgeHeap heap neighborsAddress degreesAddress neighbors degrees target count) := by
  apply BlockReturns.function rfl rfl
  change BlockReturns program _ (addEdgeBody ++ []) heap _ _
  apply (addEdge_block _ heap source target adjacencyAddress degreesAddress neighborsAddress
    adjacency degrees neighbors count rfl rfl rfl rfl adjacencyFound neighborsFound degreesFound
    neighborsLookup countLookup).returns
  exact fun _ _ => .next rfl .refl

end solvers.«79».graph
