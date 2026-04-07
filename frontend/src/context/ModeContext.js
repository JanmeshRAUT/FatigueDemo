import React, { createContext, useContext } from 'react';

const ModeContext = createContext();

export const useModeContext = () => {
    const context = useContext(ModeContext);
    if (!context) {
        console.warn("[ModeContext] Context not found");
    }
    return context;
};

export const ModeProvider = ({ children, selectedMode }) => {
    const value = {
        selectedMode, // "standard" or "vehicle"
        isStandardMode: selectedMode === "standard",
        isVehicleMode: selectedMode === "vehicle"
    };

    return (
        <ModeContext.Provider value={value}>
            {children}
        </ModeContext.Provider>
    );
};
