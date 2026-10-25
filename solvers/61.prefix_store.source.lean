import solvers.«61.world.source»
import solvers.«61.tables.source»

set_option Elab.async false

namespace solvers.«61».prefix_setup
open FormalSW.PyAst ProjectEulerSolutions
open data.«project-euler-statements».data.py.project_euler_statements.p61.table
  (lookup put encode lookup_encode encode_put lookup_put_self lookup_put_other)

abbrev NumberStore := data.«project-euler-statements».data.py.project_euler_statements.p61.table.Store

/-- An outer prefix dictionary preserves its caller and owns ordered bucket views for each processed type. -/
structure Store (base heap : Mutable.Heap) (processed : List Int) where
  entries : List (Int × Nat)
  extension : ∃ extra, heap = base ++ extra
  table : heap[base.length]? = some (.dict (encode entries))
  children : ∀ kind ∈ processed, ∃ address, base.length < address ∧ lookup entries kind = some address ∧
    Nonempty (build_prefix_map.Buckets heap address (PolygonalFamilies.family kind))

def Store.initial (base : Mutable.Heap) : Store base (base ++ [.dict []]) [] where
  entries := []
  extension := ⟨[.dict []], rfl⟩
  table := by simp [encode]
  children := by simp

variable {base heap : Mutable.Heap} {processed : List Int}

private theorem extension_set (base heap : Mutable.Heap)
    (extension : ∃ extra, heap = base ++ extra) (address : Nat) (outside : base.length ≤ address)
    (object : Mutable.Object) : ∃ extra, heap.set address object = base ++ extra := by
  obtain ⟨extra, rfl⟩ := extension
  exact ⟨extra.set (address - base.length) object, List.set_append_right address object outside⟩

def write (base after : Mutable.Heap) (entries : List (Int × Nat)) (kind : Int) (address : Nat) : Mutable.Heap :=
  after.set base.length (.dict (encode (put entries kind address)))

def Store.insert_child (store : Store base heap processed) (kind : Int)
    (child : build_prefix_map.Store heap after (PolygonalFamilies.family kind)) :
    Store base (write base after store.entries kind heap.length) (processed ++ [kind]) where
  entries := put store.entries kind heap.length
  extension := by
    apply extension_set base after ?_ base.length (by rfl)
    obtain ⟨old, parentEq⟩ := store.extension
    obtain ⟨extra, childEq⟩ := child.extension
    exact ⟨old ++ extra, by rw [childEq, parentEq, List.append_assoc]⟩
  table := by
    apply List.getElem?_set_self
    have inside := (List.getElem?_eq_some_iff.mp store.table).1
    have grown : heap.length ≤ after.length := by obtain ⟨extra, rfl⟩ := child.extension; simp
    omega
  children := by
    intro query member
    have inside := (List.getElem?_eq_some_iff.mp store.table).1
    by_cases same : query = kind
    · subst query
      exact ⟨heap.length, inside, lookup_put_self _ _ _, ⟨child.view.set_before base.length inside _⟩⟩
    · have old : query ∈ processed := by simpa [same] using member
      obtain ⟨address, outside, looked, ⟨view⟩⟩ := store.children query old
      obtain ⟨extra, childEq⟩ := child.extension
      have kept : build_prefix_map.Buckets after address (PolygonalFamilies.family query) := by
        rw [childEq]
        exact view.append extra
      refine ⟨address, outside, ?_, ⟨kept.set_before base.length outside _⟩⟩
      rw [lookup_put_other _ _ _ _ (Ne.symm same), looked]

/-- Capturing the completed dictionary supplies every recursive call's prefix lookup invariant. -/
theorem Store.world (store : Store base heap processed) (maxSides : Int)
    (covered : ∀ kind, 3 ≤ kind ∧ kind ≤ maxSides → kind ∈ processed) :
    Nonempty (dfs.World (heap ++ [.list [.ref base.length]]) heap.length maxSides) := by
  refine ⟨{
    table := base.length
    target := fun kind => (lookup store.entries kind).getD 0
    entries := encode store.entries
    cellFound := by simp
    tableFound := HeapCollection.prefix_found _ _ _ _ store.table
    looked := ?_
    buckets := ?_
  }⟩
  · intro kind valid
    obtain ⟨address, _, looked, _⟩ := store.children kind (covered kind valid)
    simp only [lookup_encode, looked, Option.map_some, Option.getD_some]
  · intro kind valid
    obtain ⟨address, _, looked, ⟨view⟩⟩ := store.children kind (covered kind valid)
    exact ⟨by simpa only [looked, Option.getD_some] using view.append [.list [.ref base.length]]⟩

end solvers.«61».prefix_setup
