import solvers.«61.base.source»
import solvers.«61.prefixes.source»
import ProjectEulerSolutions.Lib.HeapCollection

set_option Elab.async false

namespace solvers.«61».dfs
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
private abbrev program := Solvers.P61Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)

def nextValues : Expr := .call (.attribute
  (.subscript (.subscript (.name "__formalsw_cell_solve_pref_map") (.constant (.int 0))) (.name "t")) "get")
  [.name "suffix", .list []]

/-- Every required type has a complete, ordered prefix table in the captured dictionary. -/
structure World (heap : Mutable.Heap) (prefixCell : Nat) (maxSides : Int) where
  table : Nat
  target : Int → Nat
  entries : List (Constant × Mutable.Value)
  cellFound : heap[prefixCell]? = some (.list [.ref table])
  tableFound : heap[table]? = some (.dict entries)
  looked : ∀ kind, 3 ≤ kind ∧ kind ≤ maxSides →
    ScalarDict.lookup entries (.int kind) = some (.ref (target kind))
  buckets : ∀ kind, 3 ≤ kind ∧ kind ≤ maxSides →
    Nonempty (build_prefix_map.Buckets heap (target kind) (PolygonalFamilies.family kind))

variable {heap : Mutable.Heap} {prefixCell : Nat} {maxSides : Int}

def World.append (world : World heap prefixCell maxSides) (extra : Mutable.Heap) :
    World (heap ++ extra) prefixCell maxSides where
  table := world.table
  target := world.target
  entries := world.entries
  cellFound := HeapCollection.prefix_found _ _ _ _ world.cellFound
  tableFound := HeapCollection.prefix_found _ _ _ _ world.tableFound
  looked := world.looked
  buckets := by
    intro kind valid
    obtain ⟨view⟩ := world.buckets kind valid
    exact ⟨view.append extra⟩

/-- The real `.get(suffix, [])` allocates its default on hits as well as misses. -/
theorem next_values_evaluate (world : World heap prefixCell maxSides) (kind suffix : Int)
    (valid : 3 ≤ kind ∧ kind ≤ maxSides) (env : HeapCalls.Env)
    (getCell : env "__formalsw_cell_solve_pref_map" = some (.ref prefixCell))
    (getKind : env "t" = some (number kind)) (getSuffix : env "suffix" = some (number suffix))
    (globals : Mutable.Globals := []) :
    ∃ result, Evaluates program env nextValues heap (.ref result) (heap ++ [.list []]) globals ∧
      (heap ++ [.list []])[result]? =
        some (.list (numbers (build_prefix_map.bucket (PolygonalFamilies.family kind) suffix))) := by
  let target := world.target kind
  have table := world.tableFound
  have looked := world.looked kind valid
  obtain ⟨buckets⟩ := world.buckets kind valid
  have cell : Evaluates program env
      (.subscript (.name "__formalsw_cell_solve_pref_map") (.constant (.int 0))) heap
      (.ref world.table) heap globals := by
    apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
      (by intros; intro impossible; cases impossible) (Evaluates.lookup getCell) Evaluates.constant
    exact Mutable.getItem_list _ _ 0 [.ref world.table] world.cellFound (by simp)
  have receiver : Evaluates program env
      (.subscript (.subscript (.name "__formalsw_cell_solve_pref_map") (.constant (.int 0))) (.name "t"))
      heap (.ref target) heap globals := by
    apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
      (by intros; intro impossible; cases impossible) cell (Evaluates.lookup getKind)
    exact Mutable.getItem_dictionary _ _ _ table (.int kind) looked
  let extended := buckets.append [.list []]
  have sameEntries : extended.entries = buckets.entries := rfl
  cases selected : build_prefix_map.lookup buckets.entries suffix with
  | none =>
      refine ⟨heap.length, ?_, ?_⟩
      · apply Evaluates.dictionary_get (afterReceiver := heap) (afterArguments := heap ++ [.list []])
          (values := [number suffix, .ref heap.length])
          (by intro name impossible; cases impossible) receiver buckets.table
          (.cons (Evaluates.lookup getSuffix) (.cons (Evaluates.list .nil) .nil))
        change Mutable.dictionaryGet target (number suffix) (.ref heap.length) (heap ++ [.list []]) = _
        rw [Mutable.dictionaryGet_scalar _ _ _ extended.table (.int suffix) (.ref heap.length),
          build_prefix_map.lookup_encode, sameEntries, selected]
        rfl
      · rw [buckets.missing suffix selected]
        simp [numbers]
  | some result =>
      refine ⟨result, ?_, ?_⟩
      · apply Evaluates.dictionary_get (afterReceiver := heap) (afterArguments := heap ++ [.list []])
          (values := [number suffix, .ref heap.length])
          (by intro name impossible; cases impossible) receiver buckets.table
          (.cons (Evaluates.lookup getSuffix) (.cons (Evaluates.list .nil) .nil))
        change Mutable.dictionaryGet target (number suffix) (.ref heap.length) (heap ++ [.list []]) = _
        rw [Mutable.dictionaryGet_scalar _ _ _ extended.table (.int suffix) (.ref heap.length),
          build_prefix_map.lookup_encode, sameEntries, selected]
        rfl
      · exact HeapCollection.prefix_found _ _ _ _ (buckets.present suffix result selected).2

end solvers.«61».dfs
