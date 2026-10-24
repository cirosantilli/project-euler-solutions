import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12647_small : List Nat := []
theorem row12647_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12647 row12647_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12647_block_005 : List Nat := []
theorem row12647_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12647 row12647_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12647_block_006 : List Nat := []
theorem row12647_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12647 row12647_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12647_block_007 : List Nat := [12899]
def plan12647_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 31), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), (.firstFactor 13), .first, (.secondFactor 17), (.firstFactor 19), (.firstFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12647_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12647 row12647_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12647_block_007
  · decide
  · rfl

def row12647_block_008 : List Nat := []
theorem row12647_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12647 row12647_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12647_block_009 : List Nat := []
theorem row12647_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12647 row12647_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12647_block_010 : List Nat := []
theorem row12647_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12647 row12647_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12647_block_011 : List Nat := []
theorem row12647_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12647 row12647_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12647_block_012 : List Nat := []
theorem row12647_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12647 row12647_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12647_block_013 : List Nat := []
theorem row12647_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12647 row12647_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12647_large_left_left : List Nat := row12647_small ++ row12647_block_005
theorem row12647_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12647 row12647_large_left_left :=
  Budget.RowUpperBound.branch row12647_small_covers row12647_block_005_covers

def row12647_large_left_right_right : List Nat := row12647_block_007 ++ row12647_block_008
theorem row12647_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12647 row12647_large_left_right_right :=
  Budget.RowUpperBound.branch row12647_block_007_covers row12647_block_008_covers

def row12647_large_left_right : List Nat := row12647_block_006 ++ row12647_large_left_right_right
theorem row12647_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12647 row12647_large_left_right :=
  Budget.RowUpperBound.branch row12647_block_006_covers row12647_large_left_right_right_covers

def row12647_large_left : List Nat := row12647_large_left_left ++ row12647_large_left_right
theorem row12647_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12647 row12647_large_left :=
  Budget.RowUpperBound.branch row12647_large_left_left_covers row12647_large_left_right_covers

def row12647_large_right_left : List Nat := row12647_block_009 ++ row12647_block_010
theorem row12647_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12647 row12647_large_right_left :=
  Budget.RowUpperBound.branch row12647_block_009_covers row12647_block_010_covers

def row12647_large_right_right_right : List Nat := row12647_block_012 ++ row12647_block_013
theorem row12647_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12647 row12647_large_right_right_right :=
  Budget.RowUpperBound.branch row12647_block_012_covers row12647_block_013_covers

def row12647_large_right_right : List Nat := row12647_block_011 ++ row12647_large_right_right_right
theorem row12647_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12647 row12647_large_right_right :=
  Budget.RowUpperBound.branch row12647_block_011_covers row12647_large_right_right_right_covers

def row12647_large_right : List Nat := row12647_large_right_left ++ row12647_large_right_right
theorem row12647_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12647 row12647_large_right :=
  Budget.RowUpperBound.branch row12647_large_right_left_covers row12647_large_right_right_covers

def row12647_large : List Nat := row12647_large_left ++ row12647_large_right
theorem row12647_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12647 row12647_large :=
  Budget.RowUpperBound.branch row12647_large_left_covers row12647_large_right_covers

def row12647 : List Nat := row12647_large
theorem row12647_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12647 row12647 := row12647_large_covers

def row12653_small : List Nat := []
theorem row12653_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12653 row12653_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12653_block_005 : List Nat := []
theorem row12653_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12653 row12653_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12653_block_006 : List Nat := []
theorem row12653_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12653 row12653_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12653_block_007 : List Nat := [13229]
def plan12653_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, (.secondFactor 13), (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, .possible, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 31), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12653_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12653 row12653_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12653_block_007
  · decide
  · rfl

def row12653_block_008 : List Nat := []
theorem row12653_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12653 row12653_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12653_block_009 : List Nat := []
theorem row12653_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12653 row12653_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12653_block_010 : List Nat := []
theorem row12653_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12653 row12653_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12653_block_011 : List Nat := []
theorem row12653_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12653 row12653_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12653_block_012 : List Nat := []
theorem row12653_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12653 row12653_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12653_block_013 : List Nat := []
theorem row12653_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12653 row12653_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12653_large_left_left : List Nat := row12653_small ++ row12653_block_005
theorem row12653_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12653 row12653_large_left_left :=
  Budget.RowUpperBound.branch row12653_small_covers row12653_block_005_covers

def row12653_large_left_right_right : List Nat := row12653_block_007 ++ row12653_block_008
theorem row12653_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12653 row12653_large_left_right_right :=
  Budget.RowUpperBound.branch row12653_block_007_covers row12653_block_008_covers

def row12653_large_left_right : List Nat := row12653_block_006 ++ row12653_large_left_right_right
theorem row12653_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12653 row12653_large_left_right :=
  Budget.RowUpperBound.branch row12653_block_006_covers row12653_large_left_right_right_covers

def row12653_large_left : List Nat := row12653_large_left_left ++ row12653_large_left_right
theorem row12653_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12653 row12653_large_left :=
  Budget.RowUpperBound.branch row12653_large_left_left_covers row12653_large_left_right_covers

def row12653_large_right_left : List Nat := row12653_block_009 ++ row12653_block_010
theorem row12653_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12653 row12653_large_right_left :=
  Budget.RowUpperBound.branch row12653_block_009_covers row12653_block_010_covers

def row12653_large_right_right_right : List Nat := row12653_block_012 ++ row12653_block_013
theorem row12653_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12653 row12653_large_right_right_right :=
  Budget.RowUpperBound.branch row12653_block_012_covers row12653_block_013_covers

def row12653_large_right_right : List Nat := row12653_block_011 ++ row12653_large_right_right_right
theorem row12653_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12653 row12653_large_right_right :=
  Budget.RowUpperBound.branch row12653_block_011_covers row12653_large_right_right_right_covers

def row12653_large_right : List Nat := row12653_large_right_left ++ row12653_large_right_right
theorem row12653_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12653 row12653_large_right :=
  Budget.RowUpperBound.branch row12653_large_right_left_covers row12653_large_right_right_covers

def row12653_large : List Nat := row12653_large_left ++ row12653_large_right
theorem row12653_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12653 row12653_large :=
  Budget.RowUpperBound.branch row12653_large_left_covers row12653_large_right_covers

def row12653 : List Nat := row12653_large
theorem row12653_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12653 row12653 := row12653_large_covers

def row12659_small : List Nat := []
theorem row12659_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12659 row12659_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12659_block_005 : List Nat := []
theorem row12659_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12659 row12659_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12659_block_006 : List Nat := []
theorem row12659_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12659 row12659_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12659_block_007 : List Nat := [12821]
def plan12659_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .possible, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, (.secondFactor 7), (.secondFactor 29), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 13), (.secondFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12659_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12659 row12659_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12659_block_007
  · decide
  · rfl

def row12659_block_008 : List Nat := []
theorem row12659_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12659 row12659_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12659_block_009 : List Nat := []
theorem row12659_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12659 row12659_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12659_block_010 : List Nat := []
theorem row12659_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12659 row12659_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12659_block_011 : List Nat := []
theorem row12659_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12659 row12659_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12659_block_012 : List Nat := []
theorem row12659_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12659 row12659_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12659_block_013 : List Nat := []
theorem row12659_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12659 row12659_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12659_large_left_left : List Nat := row12659_small ++ row12659_block_005
theorem row12659_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12659 row12659_large_left_left :=
  Budget.RowUpperBound.branch row12659_small_covers row12659_block_005_covers

def row12659_large_left_right_right : List Nat := row12659_block_007 ++ row12659_block_008
theorem row12659_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12659 row12659_large_left_right_right :=
  Budget.RowUpperBound.branch row12659_block_007_covers row12659_block_008_covers

def row12659_large_left_right : List Nat := row12659_block_006 ++ row12659_large_left_right_right
theorem row12659_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12659 row12659_large_left_right :=
  Budget.RowUpperBound.branch row12659_block_006_covers row12659_large_left_right_right_covers

def row12659_large_left : List Nat := row12659_large_left_left ++ row12659_large_left_right
theorem row12659_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12659 row12659_large_left :=
  Budget.RowUpperBound.branch row12659_large_left_left_covers row12659_large_left_right_covers

def row12659_large_right_left : List Nat := row12659_block_009 ++ row12659_block_010
theorem row12659_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12659 row12659_large_right_left :=
  Budget.RowUpperBound.branch row12659_block_009_covers row12659_block_010_covers

def row12659_large_right_right_right : List Nat := row12659_block_012 ++ row12659_block_013
theorem row12659_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12659 row12659_large_right_right_right :=
  Budget.RowUpperBound.branch row12659_block_012_covers row12659_block_013_covers

def row12659_large_right_right : List Nat := row12659_block_011 ++ row12659_large_right_right_right
theorem row12659_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12659 row12659_large_right_right :=
  Budget.RowUpperBound.branch row12659_block_011_covers row12659_large_right_right_right_covers

def row12659_large_right : List Nat := row12659_large_right_left ++ row12659_large_right_right
theorem row12659_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12659 row12659_large_right :=
  Budget.RowUpperBound.branch row12659_large_right_left_covers row12659_large_right_right_covers

def row12659_large : List Nat := row12659_large_left ++ row12659_large_right
theorem row12659_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12659 row12659_large :=
  Budget.RowUpperBound.branch row12659_large_left_covers row12659_large_right_covers

def row12659 : List Nat := row12659_large
theorem row12659_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12659 row12659 := row12659_large_covers

def row12671_small : List Nat := []
theorem row12671_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12671 row12671_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12671_block_005 : List Nat := []
theorem row12671_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12671 row12671_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12671_block_006 : List Nat := []
theorem row12671_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12671 row12671_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12671_block_007 : List Nat := []
def plan12671_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .second, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 23), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 7), .first, .second, .first, .first, .first, .first, .first, (.secondFactor 37), (.firstFactor 23), .first, .first, (.firstFactor 17), .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12671_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12671 row12671_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12671_block_007
  · decide
  · rfl

def row12671_block_008 : List Nat := []
theorem row12671_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12671 row12671_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12671_block_009 : List Nat := []
theorem row12671_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12671 row12671_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12671_block_010 : List Nat := []
theorem row12671_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12671 row12671_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12671_block_011 : List Nat := []
theorem row12671_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12671 row12671_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12671_block_012 : List Nat := []
theorem row12671_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12671 row12671_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12671_block_013 : List Nat := []
theorem row12671_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12671 row12671_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12671_large_left_left : List Nat := row12671_small ++ row12671_block_005
theorem row12671_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12671 row12671_large_left_left :=
  Budget.RowUpperBound.branch row12671_small_covers row12671_block_005_covers

def row12671_large_left_right_right : List Nat := row12671_block_007 ++ row12671_block_008
theorem row12671_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12671 row12671_large_left_right_right :=
  Budget.RowUpperBound.branch row12671_block_007_covers row12671_block_008_covers

def row12671_large_left_right : List Nat := row12671_block_006 ++ row12671_large_left_right_right
theorem row12671_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12671 row12671_large_left_right :=
  Budget.RowUpperBound.branch row12671_block_006_covers row12671_large_left_right_right_covers

def row12671_large_left : List Nat := row12671_large_left_left ++ row12671_large_left_right
theorem row12671_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12671 row12671_large_left :=
  Budget.RowUpperBound.branch row12671_large_left_left_covers row12671_large_left_right_covers

def row12671_large_right_left : List Nat := row12671_block_009 ++ row12671_block_010
theorem row12671_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12671 row12671_large_right_left :=
  Budget.RowUpperBound.branch row12671_block_009_covers row12671_block_010_covers

def row12671_large_right_right_right : List Nat := row12671_block_012 ++ row12671_block_013
theorem row12671_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12671 row12671_large_right_right_right :=
  Budget.RowUpperBound.branch row12671_block_012_covers row12671_block_013_covers

def row12671_large_right_right : List Nat := row12671_block_011 ++ row12671_large_right_right_right
theorem row12671_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12671 row12671_large_right_right :=
  Budget.RowUpperBound.branch row12671_block_011_covers row12671_large_right_right_right_covers

def row12671_large_right : List Nat := row12671_large_right_left ++ row12671_large_right_right
theorem row12671_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12671 row12671_large_right :=
  Budget.RowUpperBound.branch row12671_large_right_left_covers row12671_large_right_right_covers

def row12671_large : List Nat := row12671_large_left ++ row12671_large_right
theorem row12671_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12671 row12671_large :=
  Budget.RowUpperBound.branch row12671_large_left_covers row12671_large_right_covers

def row12671 : List Nat := row12671_large
theorem row12671_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12671 row12671 := row12671_large_covers

end ProjectEulerSolutions.PrimePairLarge
