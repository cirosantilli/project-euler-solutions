import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12689_small : List Nat := []
theorem row12689_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12689 row12689_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12689_block_005 : List Nat := []
theorem row12689_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12689 row12689_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12689_block_006 : List Nat := []
theorem row12689_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12689 row12689_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12689_block_007 : List Nat := []
def plan12689_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), (.secondFactor 13), .first, (.firstFactor 7), .second, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 13), (.secondFactor 17), .first, (.secondFactor 37), .first, .first, (.firstFactor 7), (.firstFactor 19), (.firstFactor 31), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12689_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12689 row12689_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12689_block_007
  · decide
  · rfl

def row12689_block_008 : List Nat := []
theorem row12689_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12689 row12689_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12689_block_009 : List Nat := []
theorem row12689_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12689 row12689_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12689_block_010 : List Nat := []
theorem row12689_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12689 row12689_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12689_block_011 : List Nat := []
theorem row12689_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12689 row12689_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12689_block_012 : List Nat := []
theorem row12689_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12689 row12689_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12689_block_013 : List Nat := []
theorem row12689_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12689 row12689_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12689_large_left_left : List Nat := row12689_small ++ row12689_block_005
theorem row12689_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12689 row12689_large_left_left :=
  Budget.RowUpperBound.branch row12689_small_covers row12689_block_005_covers

def row12689_large_left_right_right : List Nat := row12689_block_007 ++ row12689_block_008
theorem row12689_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12689 row12689_large_left_right_right :=
  Budget.RowUpperBound.branch row12689_block_007_covers row12689_block_008_covers

def row12689_large_left_right : List Nat := row12689_block_006 ++ row12689_large_left_right_right
theorem row12689_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12689 row12689_large_left_right :=
  Budget.RowUpperBound.branch row12689_block_006_covers row12689_large_left_right_right_covers

def row12689_large_left : List Nat := row12689_large_left_left ++ row12689_large_left_right
theorem row12689_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12689 row12689_large_left :=
  Budget.RowUpperBound.branch row12689_large_left_left_covers row12689_large_left_right_covers

def row12689_large_right_left : List Nat := row12689_block_009 ++ row12689_block_010
theorem row12689_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12689 row12689_large_right_left :=
  Budget.RowUpperBound.branch row12689_block_009_covers row12689_block_010_covers

def row12689_large_right_right_right : List Nat := row12689_block_012 ++ row12689_block_013
theorem row12689_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12689 row12689_large_right_right_right :=
  Budget.RowUpperBound.branch row12689_block_012_covers row12689_block_013_covers

def row12689_large_right_right : List Nat := row12689_block_011 ++ row12689_large_right_right_right
theorem row12689_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12689 row12689_large_right_right :=
  Budget.RowUpperBound.branch row12689_block_011_covers row12689_large_right_right_right_covers

def row12689_large_right : List Nat := row12689_large_right_left ++ row12689_large_right_right
theorem row12689_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12689 row12689_large_right :=
  Budget.RowUpperBound.branch row12689_large_right_left_covers row12689_large_right_right_covers

def row12689_large : List Nat := row12689_large_left ++ row12689_large_right
theorem row12689_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12689 row12689_large :=
  Budget.RowUpperBound.branch row12689_large_left_covers row12689_large_right_covers

def row12689 : List Nat := row12689_large
theorem row12689_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12689 row12689 := row12689_large_covers

def row12697_small : List Nat := []
theorem row12697_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12697 row12697_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12697_block_005 : List Nat := []
theorem row12697_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12697 row12697_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12697_block_006 : List Nat := []
theorem row12697_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12697 row12697_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12697_block_007 : List Nat := []
def plan12697_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 37), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 37), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 31), (.firstFactor 37), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, .second, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 29), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second, (.firstFactor 7), (.secondFactor 37), .first, .first, .second, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12697_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12697 row12697_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12697_block_007
  · decide
  · rfl

def row12697_block_008 : List Nat := []
theorem row12697_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12697 row12697_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12697_block_009 : List Nat := []
theorem row12697_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12697 row12697_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12697_block_010 : List Nat := []
theorem row12697_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12697 row12697_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12697_block_011 : List Nat := []
theorem row12697_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12697 row12697_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12697_block_012 : List Nat := []
theorem row12697_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12697 row12697_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12697_block_013 : List Nat := []
theorem row12697_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12697 row12697_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12697_large_left_left : List Nat := row12697_small ++ row12697_block_005
theorem row12697_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12697 row12697_large_left_left :=
  Budget.RowUpperBound.branch row12697_small_covers row12697_block_005_covers

def row12697_large_left_right_right : List Nat := row12697_block_007 ++ row12697_block_008
theorem row12697_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12697 row12697_large_left_right_right :=
  Budget.RowUpperBound.branch row12697_block_007_covers row12697_block_008_covers

def row12697_large_left_right : List Nat := row12697_block_006 ++ row12697_large_left_right_right
theorem row12697_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12697 row12697_large_left_right :=
  Budget.RowUpperBound.branch row12697_block_006_covers row12697_large_left_right_right_covers

def row12697_large_left : List Nat := row12697_large_left_left ++ row12697_large_left_right
theorem row12697_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12697 row12697_large_left :=
  Budget.RowUpperBound.branch row12697_large_left_left_covers row12697_large_left_right_covers

def row12697_large_right_left : List Nat := row12697_block_009 ++ row12697_block_010
theorem row12697_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12697 row12697_large_right_left :=
  Budget.RowUpperBound.branch row12697_block_009_covers row12697_block_010_covers

def row12697_large_right_right_right : List Nat := row12697_block_012 ++ row12697_block_013
theorem row12697_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12697 row12697_large_right_right_right :=
  Budget.RowUpperBound.branch row12697_block_012_covers row12697_block_013_covers

def row12697_large_right_right : List Nat := row12697_block_011 ++ row12697_large_right_right_right
theorem row12697_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12697 row12697_large_right_right :=
  Budget.RowUpperBound.branch row12697_block_011_covers row12697_large_right_right_right_covers

def row12697_large_right : List Nat := row12697_large_right_left ++ row12697_large_right_right
theorem row12697_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12697 row12697_large_right :=
  Budget.RowUpperBound.branch row12697_large_right_left_covers row12697_large_right_right_covers

def row12697_large : List Nat := row12697_large_left ++ row12697_large_right
theorem row12697_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12697 row12697_large :=
  Budget.RowUpperBound.branch row12697_large_left_covers row12697_large_right_covers

def row12697 : List Nat := row12697_large
theorem row12697_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12697 row12697 := row12697_large_covers

def row12703_small : List Nat := []
theorem row12703_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12703 row12703_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12703_block_005 : List Nat := []
theorem row12703_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12703 row12703_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12703_block_006 : List Nat := []
theorem row12703_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12703 row12703_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12703_block_007 : List Nat := [13171]
def plan12703_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .second, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 11), (.secondFactor 19), .first, (.secondFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12703_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12703 row12703_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12703_block_007
  · decide
  · rfl

def row12703_block_008 : List Nat := []
theorem row12703_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12703 row12703_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12703_block_009 : List Nat := []
theorem row12703_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12703 row12703_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12703_block_010 : List Nat := []
theorem row12703_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12703 row12703_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12703_block_011 : List Nat := []
theorem row12703_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12703 row12703_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12703_block_012 : List Nat := []
theorem row12703_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12703 row12703_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12703_block_013 : List Nat := []
theorem row12703_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12703 row12703_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12703_large_left_left : List Nat := row12703_small ++ row12703_block_005
theorem row12703_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12703 row12703_large_left_left :=
  Budget.RowUpperBound.branch row12703_small_covers row12703_block_005_covers

def row12703_large_left_right_right : List Nat := row12703_block_007 ++ row12703_block_008
theorem row12703_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12703 row12703_large_left_right_right :=
  Budget.RowUpperBound.branch row12703_block_007_covers row12703_block_008_covers

def row12703_large_left_right : List Nat := row12703_block_006 ++ row12703_large_left_right_right
theorem row12703_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12703 row12703_large_left_right :=
  Budget.RowUpperBound.branch row12703_block_006_covers row12703_large_left_right_right_covers

def row12703_large_left : List Nat := row12703_large_left_left ++ row12703_large_left_right
theorem row12703_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12703 row12703_large_left :=
  Budget.RowUpperBound.branch row12703_large_left_left_covers row12703_large_left_right_covers

def row12703_large_right_left : List Nat := row12703_block_009 ++ row12703_block_010
theorem row12703_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12703 row12703_large_right_left :=
  Budget.RowUpperBound.branch row12703_block_009_covers row12703_block_010_covers

def row12703_large_right_right_right : List Nat := row12703_block_012 ++ row12703_block_013
theorem row12703_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12703 row12703_large_right_right_right :=
  Budget.RowUpperBound.branch row12703_block_012_covers row12703_block_013_covers

def row12703_large_right_right : List Nat := row12703_block_011 ++ row12703_large_right_right_right
theorem row12703_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12703 row12703_large_right_right :=
  Budget.RowUpperBound.branch row12703_block_011_covers row12703_large_right_right_right_covers

def row12703_large_right : List Nat := row12703_large_right_left ++ row12703_large_right_right
theorem row12703_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12703 row12703_large_right :=
  Budget.RowUpperBound.branch row12703_large_right_left_covers row12703_large_right_right_covers

def row12703_large : List Nat := row12703_large_left ++ row12703_large_right
theorem row12703_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12703 row12703_large :=
  Budget.RowUpperBound.branch row12703_large_left_covers row12703_large_right_covers

def row12703 : List Nat := row12703_large
theorem row12703_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12703 row12703 := row12703_large_covers

def row12713_small : List Nat := []
theorem row12713_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12713 row12713_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12713_block_005 : List Nat := []
theorem row12713_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12713 row12713_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12713_block_006 : List Nat := []
theorem row12713_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12713 row12713_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12713_block_007 : List Nat := [13037]
def plan12713_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 31), .first, .first, .first, .first, .first, (.secondFactor 13), .second, .first, (.firstFactor 11), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .possible, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.secondFactor 19), (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, (.secondFactor 19), .first, (.firstFactor 37), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12713_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12713 row12713_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12713_block_007
  · decide
  · rfl

def row12713_block_008 : List Nat := []
theorem row12713_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12713 row12713_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12713_block_009 : List Nat := []
theorem row12713_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12713 row12713_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12713_block_010 : List Nat := []
theorem row12713_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12713 row12713_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12713_block_011 : List Nat := []
theorem row12713_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12713 row12713_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12713_block_012 : List Nat := []
theorem row12713_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12713 row12713_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12713_block_013 : List Nat := []
theorem row12713_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12713 row12713_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12713_large_left_left : List Nat := row12713_small ++ row12713_block_005
theorem row12713_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12713 row12713_large_left_left :=
  Budget.RowUpperBound.branch row12713_small_covers row12713_block_005_covers

def row12713_large_left_right_right : List Nat := row12713_block_007 ++ row12713_block_008
theorem row12713_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12713 row12713_large_left_right_right :=
  Budget.RowUpperBound.branch row12713_block_007_covers row12713_block_008_covers

def row12713_large_left_right : List Nat := row12713_block_006 ++ row12713_large_left_right_right
theorem row12713_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12713 row12713_large_left_right :=
  Budget.RowUpperBound.branch row12713_block_006_covers row12713_large_left_right_right_covers

def row12713_large_left : List Nat := row12713_large_left_left ++ row12713_large_left_right
theorem row12713_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12713 row12713_large_left :=
  Budget.RowUpperBound.branch row12713_large_left_left_covers row12713_large_left_right_covers

def row12713_large_right_left : List Nat := row12713_block_009 ++ row12713_block_010
theorem row12713_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12713 row12713_large_right_left :=
  Budget.RowUpperBound.branch row12713_block_009_covers row12713_block_010_covers

def row12713_large_right_right_right : List Nat := row12713_block_012 ++ row12713_block_013
theorem row12713_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12713 row12713_large_right_right_right :=
  Budget.RowUpperBound.branch row12713_block_012_covers row12713_block_013_covers

def row12713_large_right_right : List Nat := row12713_block_011 ++ row12713_large_right_right_right
theorem row12713_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12713 row12713_large_right_right :=
  Budget.RowUpperBound.branch row12713_block_011_covers row12713_large_right_right_right_covers

def row12713_large_right : List Nat := row12713_large_right_left ++ row12713_large_right_right
theorem row12713_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12713 row12713_large_right :=
  Budget.RowUpperBound.branch row12713_large_right_left_covers row12713_large_right_right_covers

def row12713_large : List Nat := row12713_large_left ++ row12713_large_right
theorem row12713_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12713 row12713_large :=
  Budget.RowUpperBound.branch row12713_large_left_covers row12713_large_right_covers

def row12713 : List Nat := row12713_large
theorem row12713_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12713 row12713 := row12713_large_covers

end ProjectEulerSolutions.PrimePairLarge
