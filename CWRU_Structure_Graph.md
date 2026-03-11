# CWRU Dataset Folder Structure Graph

## Complete Hierarchy Visualization

```mermaid
graph TD
    A["CWRU-dataset"] --> B["12k_Drive_End_Bearing_Fault_Data"]
    A --> C["12k_Fan_End_Bearing_Fault_Data"]
    A --> D["48k_Drive_End_Bearing_Fault_Data"]
    A --> E["Normal"]
    
    B --> B1["B"]
    B --> B2["IR"]
    B --> B3["OR"]
    
    B1 --> B1a["007"]
    B1 --> B1b["014"]
    B1 --> B1c["021"]
    B1 --> B1d["028"]
    
    B2 --> B2a["007"]
    B2 --> B2b["014"]
    B2 --> B2c["021"]
    B2 --> B2d["028"]
    
    B3 --> B3a["007"]
    B3 --> B3b["014"]
    B3 --> B3c["021"]
    
    B1a --> B1a1["MAT Files"]
    B1b --> B1b1["MAT Files"]
    B1c --> B1c1["MAT Files"]
    B1d --> B1d1["MAT Files"]
    
    B2a --> B2a1["MAT Files"]
    B2b --> B2b1["MAT Files"]
    B2c --> B2c1["MAT Files"]
    B2d --> B2d1["MAT Files"]
    
    B3a --> B3a1["@12/@3/@6 subdirs"]
    B3b --> B3b1["MAT Files"]
    B3c --> B3c1["@12/@3/@6 subdirs"]
    
    C --> C1["B"]
    C --> C2["IR"]
    C --> C3["OR"]
    
    C1 --> C1a["007"]
    C1 --> C1b["014"]
    C1 --> C1c["021"]
    
    C2 --> C2a["007"]
    C2 --> C2b["014"]
    C2 --> C2c["021"]
    
    C3 --> C3a["007"]
    C3 --> C3b["014"]
    C3 --> C3c["021"]
    
    D --> D1["B"]
    D --> D2["IR"]
    D --> D3["OR"]
    
    D1 --> D1a["007"]
    D1 --> D1b["014"]
    D1 --> D1c["021"]
    
    D2 --> D2a["007"]
    D2 --> D2b["014"]
    D2 --> D2c["021"]
    
    D3 --> D3a["007"]
    D3 --> D3b["014"]
    D3 --> D3c["021"]
    
    E --> E1["MAT Files"]
    
    style A fill:#ff6b6b
    style B fill:#4ecdc4
    style C fill:#4ecdc4
    style D fill:#4ecdc4
    style E fill:#95e1d3
```

## Simplified Hierarchical View

```mermaid
graph TD
    CWRU["🗂️ CWRU-dataset"]
    
    CWRU --> D12K["📁 12k_Drive_End<br/>Bearing_Fault_Data"]
    CWRU --> F12K["📁 12k_Fan_End<br/>Bearing_Fault_Data"]
    CWRU --> D48K["📁 48k_Drive_End<br/>Bearing_Fault_Data"]
    CWRU --> NORM["📁 Normal<br/>4 MAT files"]
    
    D12K --> DB12["Bearing"]
    D12K --> DIR12["Inner Race"]
    D12K --> DOR12["Outer Race"]
    
    DB12 --> DB12L["007, 014, 021, 028"]
    DIR12 --> DIR12L["007, 014, 021, 028"]
    DOR12 --> DOR12L["007: @3/@6/@12<br/>014: MAT files<br/>021: @3/@6/@12"]
    
    F12K --> FB12["Bearing"]
    F12K --> FIR12["Inner Race"]
    F12K --> FOR12["Outer Race"]
    
    FB12 --> FB12L["007, 014, 021"]
    FIR12 --> FIR12L["007, 014, 021"]
    FOR12 --> FOR12L["007, 014, 021"]
    
    D48K --> DB48["Bearing"]
    D48K --> DIR48["Inner Race"]
    D48K --> DOR48["Outer Race"]
    
    DB48 --> DB48L["007, 014, 021"]
    DIR48 --> DIR48L["007, 014, 021"]
    DOR48 --> DOR48L["007: @3/@6/@12<br/>014: MAT files<br/>021: @3/@6/@12"]
    
    style CWRU fill:#ff6b6b,color:#fff,stroke:#333,stroke-width:3px
    style D12K fill:#4ecdc4,color:#fff,stroke:#333,stroke-width:2px
    style F12K fill:#4ecdc4,color:#fff,stroke:#333,stroke-width:2px
    style D48K fill:#4ecdc4,color:#fff,stroke:#333,stroke-width:2px
    style NORM fill:#95e1d3,color:#333,stroke:#333,stroke-width:2px
```

## Fault Type Distribution Matrix

```mermaid
graph LR
    subgraph F["Fault Types<br/>(at each load level)"]
        B["Ball Bearing"]
        IR["Inner Race"]
        OR["Outer Race"]
    end
    
    subgraph D["Data Conditions"]
        D12K["12K RPM<br/>Drive End"]
        F12K["12K RPM<br/>Fan End"]
        D48K["48K RPM<br/>Drive End"]
    end
    
    subgraph L["Load Levels<br/>per Fault Type"]
        L1["007"]
        L2["014"]
        L3["021"]
        L4["028"]
    end
    
    style F fill:#ffe66d
    style D fill:#95e1d3
    style L fill:#c7ceea
```

## File Distribution Summary

| Category | Structure | Load Levels | Fault Types | Files |
|----------|-----------|-------------|------------|-------|
| **12K Drive** | 12k_Drive_End_Bearing_Fault_Data | 4 (007,014,021,028) | 3 (B,IR,OR) | 48+ MAT |
| **12K Fan** | 12k_Fan_End_Bearing_Fault_Data | 3 (007,014,021) | 3 (B,IR,OR) | 36+ MAT |
| **48K Drive** | 48k_Drive_End_Bearing_Fault_Data | 3 (007,014,021) | 3 (B,IR,OR) | 36+ MAT |
| **Normal** | Normal | — | Healthy | 4 MAT |

## Legend

- **B** = Ball bearing fault
- **IR** = Inner race fault  
- **OR** = Outer race fault
- **007, 014, 021, 028** = Load levels (HP: horsepower)
- **@3, @6, @12** = Fault diameters in inches (for OR faults)
- **MAT files** = Raw bearing vibration data in MATLAB format

