import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12491_small : List Nat := []
theorem row12491_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12491 row12491_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12491_block_005 : List Nat := []
theorem row12491_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12491 row12491_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12491_block_006 : List Nat := []
theorem row12491_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12491 row12491_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12491_block_007 : List Nat := [12713, 13469]
def plan12491_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 19), (.secondFactor 31), .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 23), (.firstFactor 13), .first, .second, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 37), (.secondFactor 29), (.firstFactor 13), .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12491_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12491 row12491_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12491_block_007
  · decide
  · rfl

def row12491_block_008 : List Nat := []
theorem row12491_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12491 row12491_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12491_block_009 : List Nat := []
theorem row12491_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12491 row12491_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12491_block_010 : List Nat := []
theorem row12491_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12491 row12491_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12491_block_011 : List Nat := []
theorem row12491_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12491 row12491_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12491_block_012 : List Nat := []
theorem row12491_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12491 row12491_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12491_block_013 : List Nat := []
theorem row12491_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12491 row12491_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12491_large_left_left : List Nat := row12491_small ++ row12491_block_005
theorem row12491_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12491 row12491_large_left_left :=
  Budget.RowUpperBound.branch row12491_small_covers row12491_block_005_covers

def row12491_large_left_right_right : List Nat := row12491_block_007 ++ row12491_block_008
theorem row12491_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12491 row12491_large_left_right_right :=
  Budget.RowUpperBound.branch row12491_block_007_covers row12491_block_008_covers

def row12491_large_left_right : List Nat := row12491_block_006 ++ row12491_large_left_right_right
theorem row12491_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12491 row12491_large_left_right :=
  Budget.RowUpperBound.branch row12491_block_006_covers row12491_large_left_right_right_covers

def row12491_large_left : List Nat := row12491_large_left_left ++ row12491_large_left_right
theorem row12491_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12491 row12491_large_left :=
  Budget.RowUpperBound.branch row12491_large_left_left_covers row12491_large_left_right_covers

def row12491_large_right_left : List Nat := row12491_block_009 ++ row12491_block_010
theorem row12491_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12491 row12491_large_right_left :=
  Budget.RowUpperBound.branch row12491_block_009_covers row12491_block_010_covers

def row12491_large_right_right_right : List Nat := row12491_block_012 ++ row12491_block_013
theorem row12491_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12491 row12491_large_right_right_right :=
  Budget.RowUpperBound.branch row12491_block_012_covers row12491_block_013_covers

def row12491_large_right_right : List Nat := row12491_block_011 ++ row12491_large_right_right_right
theorem row12491_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12491 row12491_large_right_right :=
  Budget.RowUpperBound.branch row12491_block_011_covers row12491_large_right_right_right_covers

def row12491_large_right : List Nat := row12491_large_right_left ++ row12491_large_right_right
theorem row12491_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12491 row12491_large_right :=
  Budget.RowUpperBound.branch row12491_large_right_left_covers row12491_large_right_right_covers

def row12491_large : List Nat := row12491_large_left ++ row12491_large_right
theorem row12491_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12491 row12491_large :=
  Budget.RowUpperBound.branch row12491_large_left_covers row12491_large_right_covers

def row12491 : List Nat := row12491_large
theorem row12491_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12491 row12491 := row12491_large_covers

def row12497_small : List Nat := []
theorem row12497_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12497 row12497_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12497_block_005 : List Nat := []
theorem row12497_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12497 row12497_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12497_block_006 : List Nat := []
theorem row12497_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12497 row12497_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12497_block_007 : List Nat := [12647, 12899]
def plan12497_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 31), .possible, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), (.firstFactor 29), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .possible, .first, .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), (.secondFactor 23), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, (.secondFactor 37), .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 23), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12497_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12497 row12497_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12497_block_007
  · decide
  · rfl

def row12497_block_008 : List Nat := []
theorem row12497_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12497 row12497_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12497_block_009 : List Nat := []
theorem row12497_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12497 row12497_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12497_block_010 : List Nat := []
theorem row12497_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12497 row12497_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12497_block_011 : List Nat := []
theorem row12497_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12497 row12497_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12497_block_012 : List Nat := []
theorem row12497_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12497 row12497_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12497_block_013 : List Nat := []
theorem row12497_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12497 row12497_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12497_large_left_left : List Nat := row12497_small ++ row12497_block_005
theorem row12497_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12497 row12497_large_left_left :=
  Budget.RowUpperBound.branch row12497_small_covers row12497_block_005_covers

def row12497_large_left_right_right : List Nat := row12497_block_007 ++ row12497_block_008
theorem row12497_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12497 row12497_large_left_right_right :=
  Budget.RowUpperBound.branch row12497_block_007_covers row12497_block_008_covers

def row12497_large_left_right : List Nat := row12497_block_006 ++ row12497_large_left_right_right
theorem row12497_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12497 row12497_large_left_right :=
  Budget.RowUpperBound.branch row12497_block_006_covers row12497_large_left_right_right_covers

def row12497_large_left : List Nat := row12497_large_left_left ++ row12497_large_left_right
theorem row12497_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12497 row12497_large_left :=
  Budget.RowUpperBound.branch row12497_large_left_left_covers row12497_large_left_right_covers

def row12497_large_right_left : List Nat := row12497_block_009 ++ row12497_block_010
theorem row12497_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12497 row12497_large_right_left :=
  Budget.RowUpperBound.branch row12497_block_009_covers row12497_block_010_covers

def row12497_large_right_right_right : List Nat := row12497_block_012 ++ row12497_block_013
theorem row12497_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12497 row12497_large_right_right_right :=
  Budget.RowUpperBound.branch row12497_block_012_covers row12497_block_013_covers

def row12497_large_right_right : List Nat := row12497_block_011 ++ row12497_large_right_right_right
theorem row12497_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12497 row12497_large_right_right :=
  Budget.RowUpperBound.branch row12497_block_011_covers row12497_large_right_right_right_covers

def row12497_large_right : List Nat := row12497_large_right_left ++ row12497_large_right_right
theorem row12497_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12497 row12497_large_right :=
  Budget.RowUpperBound.branch row12497_large_right_left_covers row12497_large_right_right_covers

def row12497_large : List Nat := row12497_large_left ++ row12497_large_right
theorem row12497_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12497 row12497_large :=
  Budget.RowUpperBound.branch row12497_large_left_covers row12497_large_right_covers

def row12497 : List Nat := row12497_large
theorem row12497_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12497 row12497 := row12497_large_covers

def row12503_small : List Nat := []
theorem row12503_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12503 row12503_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12503_block_005 : List Nat := []
theorem row12503_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12503 row12503_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12503_block_006 : List Nat := []
theorem row12503_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12503 row12503_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12503_block_007 : List Nat := []
def plan12503_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 29), (.secondFactor 37), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 23), .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 13), .second, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 31), (.secondFactor 23), .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 37), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 29), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12503_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12503 row12503_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12503_block_007
  · decide
  · rfl

def row12503_block_008 : List Nat := []
theorem row12503_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12503 row12503_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12503_block_009 : List Nat := []
theorem row12503_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12503 row12503_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12503_block_010 : List Nat := []
theorem row12503_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12503 row12503_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12503_block_011 : List Nat := []
theorem row12503_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12503 row12503_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12503_block_012 : List Nat := []
theorem row12503_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12503 row12503_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12503_block_013 : List Nat := []
theorem row12503_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12503 row12503_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12503_large_left_left : List Nat := row12503_small ++ row12503_block_005
theorem row12503_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12503 row12503_large_left_left :=
  Budget.RowUpperBound.branch row12503_small_covers row12503_block_005_covers

def row12503_large_left_right_right : List Nat := row12503_block_007 ++ row12503_block_008
theorem row12503_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12503 row12503_large_left_right_right :=
  Budget.RowUpperBound.branch row12503_block_007_covers row12503_block_008_covers

def row12503_large_left_right : List Nat := row12503_block_006 ++ row12503_large_left_right_right
theorem row12503_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12503 row12503_large_left_right :=
  Budget.RowUpperBound.branch row12503_block_006_covers row12503_large_left_right_right_covers

def row12503_large_left : List Nat := row12503_large_left_left ++ row12503_large_left_right
theorem row12503_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12503 row12503_large_left :=
  Budget.RowUpperBound.branch row12503_large_left_left_covers row12503_large_left_right_covers

def row12503_large_right_left : List Nat := row12503_block_009 ++ row12503_block_010
theorem row12503_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12503 row12503_large_right_left :=
  Budget.RowUpperBound.branch row12503_block_009_covers row12503_block_010_covers

def row12503_large_right_right_right : List Nat := row12503_block_012 ++ row12503_block_013
theorem row12503_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12503 row12503_large_right_right_right :=
  Budget.RowUpperBound.branch row12503_block_012_covers row12503_block_013_covers

def row12503_large_right_right : List Nat := row12503_block_011 ++ row12503_large_right_right_right
theorem row12503_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12503 row12503_large_right_right :=
  Budget.RowUpperBound.branch row12503_block_011_covers row12503_large_right_right_right_covers

def row12503_large_right : List Nat := row12503_large_right_left ++ row12503_large_right_right
theorem row12503_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12503 row12503_large_right :=
  Budget.RowUpperBound.branch row12503_large_right_left_covers row12503_large_right_right_covers

def row12503_large : List Nat := row12503_large_left ++ row12503_large_right
theorem row12503_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12503 row12503_large :=
  Budget.RowUpperBound.branch row12503_large_left_covers row12503_large_right_covers

def row12503 : List Nat := row12503_large
theorem row12503_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12503 row12503 := row12503_large_covers

def row12511_small : List Nat := []
theorem row12511_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12511 row12511_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12511_block_005 : List Nat := []
theorem row12511_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12511 row12511_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12511_block_006 : List Nat := []
theorem row12511_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12511 row12511_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12511_block_007 : List Nat := []
def plan12511_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 19), (.firstFactor 31), .first, (.secondFactor 19), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .second, (.firstFactor 19), .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 13), .first, .second, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12511_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12511 row12511_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12511_block_007
  · decide
  · rfl

def row12511_block_008 : List Nat := []
theorem row12511_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12511 row12511_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12511_block_009 : List Nat := []
theorem row12511_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12511 row12511_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12511_block_010 : List Nat := []
theorem row12511_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12511 row12511_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12511_block_011 : List Nat := []
theorem row12511_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12511 row12511_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12511_block_012 : List Nat := []
theorem row12511_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12511 row12511_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12511_block_013 : List Nat := []
theorem row12511_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12511 row12511_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12511_large_left_left : List Nat := row12511_small ++ row12511_block_005
theorem row12511_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12511 row12511_large_left_left :=
  Budget.RowUpperBound.branch row12511_small_covers row12511_block_005_covers

def row12511_large_left_right_right : List Nat := row12511_block_007 ++ row12511_block_008
theorem row12511_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12511 row12511_large_left_right_right :=
  Budget.RowUpperBound.branch row12511_block_007_covers row12511_block_008_covers

def row12511_large_left_right : List Nat := row12511_block_006 ++ row12511_large_left_right_right
theorem row12511_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12511 row12511_large_left_right :=
  Budget.RowUpperBound.branch row12511_block_006_covers row12511_large_left_right_right_covers

def row12511_large_left : List Nat := row12511_large_left_left ++ row12511_large_left_right
theorem row12511_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12511 row12511_large_left :=
  Budget.RowUpperBound.branch row12511_large_left_left_covers row12511_large_left_right_covers

def row12511_large_right_left : List Nat := row12511_block_009 ++ row12511_block_010
theorem row12511_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12511 row12511_large_right_left :=
  Budget.RowUpperBound.branch row12511_block_009_covers row12511_block_010_covers

def row12511_large_right_right_right : List Nat := row12511_block_012 ++ row12511_block_013
theorem row12511_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12511 row12511_large_right_right_right :=
  Budget.RowUpperBound.branch row12511_block_012_covers row12511_block_013_covers

def row12511_large_right_right : List Nat := row12511_block_011 ++ row12511_large_right_right_right
theorem row12511_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12511 row12511_large_right_right :=
  Budget.RowUpperBound.branch row12511_block_011_covers row12511_large_right_right_right_covers

def row12511_large_right : List Nat := row12511_large_right_left ++ row12511_large_right_right
theorem row12511_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12511 row12511_large_right :=
  Budget.RowUpperBound.branch row12511_large_right_left_covers row12511_large_right_right_covers

def row12511_large : List Nat := row12511_large_left ++ row12511_large_right
theorem row12511_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12511 row12511_large :=
  Budget.RowUpperBound.branch row12511_large_left_covers row12511_large_right_covers

def row12511 : List Nat := row12511_large
theorem row12511_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12511 row12511 := row12511_large_covers

end ProjectEulerSolutions.PrimePairLarge
